# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

logger = init_logger(__name__)


class TalkerMTPCudaGraphWrapper:
    """
    CUDA Graph wrapper for talker_mtp (multi-token prediction).

    Captures the entire MTP pipeline:
    - Code predictor forward (residual codebooks, argmax/deterministic)
    - Embedding summation
    - Text step addition
    """

    def __init__(
        self,
        talker_model,
        talker_config,
        capture_sizes,
        device='cuda',
        enabled=True,
    ):
        self.device = device
        self.device_index = torch.device(device).index or 0
        self.capture_sizes = capture_sizes
        self.enabled = enabled

        self.talker = talker_model
        self.code_predictor = talker_model.code_predictor
        self.num_code_groups = talker_config.num_code_groups
        self.hidden_size = talker_config.hidden_size

        # Static input buffers (fixed on GPU)
        self.input_ids_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.last_id_hidden_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.bfloat16, device=device)
        self.past_hidden_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.bfloat16, device=device)
        self.text_step_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.bfloat16, device=device)

        # Static output buffers
        self.audio_codes_buf = torch.zeros(1, self.num_code_groups, dtype=torch.long, device=device)
        self.inputs_embeds_out_buf = torch.zeros(1, self.hidden_size, dtype=torch.bfloat16, device=device)

        self.graph = None
        self.captured = False

    def _mtp_forward(self):
        """The actual MTP computation to be captured."""
        # Code predictor forward — deterministic argmax, required for CUDA graph.
        audio_codes = self.code_predictor.forward_deterministic(
            layer0_code=self.input_ids_buf,
            layer0_embed=self.last_id_hidden_buf,
            last_talker_hidden=self.past_hidden_buf,
        )
        self.audio_codes_buf.copy_(audio_codes)

        # Embedding summation
        residual_ids = audio_codes[:, 1:]
        embeds = [self.last_id_hidden_buf]
        for i in range(self.num_code_groups - 1):
            emb = self.code_predictor.get_input_embeddings()[i](residual_ids[:, i : i + 1])
            embeds.append(emb)

        summed = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        result = (summed + self.text_step_buf).reshape(1, -1)
        self.inputs_embeds_out_buf.copy_(result)

    def capture(self):
        """Capture the MTP forward as CUDA graph."""
        # Warmup runs on the default stream to trigger torch.compile JIT and
        # pre-allocate all buffers before graph capture.
        for _ in range(3):
            self._mtp_forward()
        torch.cuda.synchronize(self.device)

        # Capture on a dedicated side stream so the default stream is not
        # polluted by graph memory.
        with torch.cuda.device(self.device_index):
            self.graph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # One additional warmup on the capture stream.
                self._mtp_forward()
                s.synchronize()
                with torch.cuda.graph(self.graph):
                    self._mtp_forward()

        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        """Warm up and capture the CUDA graph."""
        if not self.enabled:
            logger.info("TalkerMTPCudaGraphWrapper: disabled, skipping capture")
            return
        # Update device bookkeeping in case it was passed as a torch.device.
        self.device = device
        self.device_index = device.index or 0
        self.capture()
        logger.info("TalkerMTPCudaGraphWrapper: CUDA graph captured")

    @torch.inference_mode()
    def _talker_mtp(self, input_ids, last_id_hidden, past_hidden, text_step):
        """
        Run MTP with graph replay.

        Args: GPU tensors (already on device, from preprocess)
        Returns: (inputs_embeds_out, audio_codes) - GPU tensors
        """
        if not self.captured or self.graph is None:
            raise RuntimeError(
                "TalkerMTPCudaGraphWrapper: graph not captured — call warmup() first"
            )

        # Copy inputs into static buffers
        self.input_ids_buf.copy_(input_ids.reshape(1, 1))
        self.last_id_hidden_buf.copy_(last_id_hidden.reshape(1, 1, -1))
        self.past_hidden_buf.copy_(past_hidden.reshape(1, 1, -1))
        self.text_step_buf.copy_(text_step.reshape(1, 1, -1))

        # Replay graph
        self.graph.replay()

        # Return outputs (clone to avoid graph memory aliasing)
        return self.inputs_embeds_out_buf.clone(), self.audio_codes_buf.clone()


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes)  # Automatically uses CUDA graph if possible
    """

    DEFAULT_CAPTURE_SIZES = [2, 4, 8, 16, 25, 32, 50, 100, 150, 200, 250, 300]

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self.capture_sizes = capture_sizes or self.DEFAULT_CAPTURE_SIZES
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device = None

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        if device.type != "cuda":
            logger.info("CUDA Graph warmup skipped: device %s is not CUDA", device)
            return

        if not self.enabled:
            logger.info("CUDA Graph is disabled, skipping warmup")
            return

        if self._warmed_up:
            logger.warning("CUDA Graph already warmed up, skipping")
            return

        self._device = device
        self.decoder.eval()

        logger.info("Starting CUDA Graph warmup for %d sizes: %s", len(self.capture_sizes), self.capture_sizes)

        # Warmup runs to ensure CUDA memory is allocated
        for size in self.capture_sizes:
            dummy_codes = torch.zeros(
                1,
                self.num_quantizers,
                size,
                dtype=dtype,
                device=device,
            )
            with torch.no_grad():
                _ = self.decoder(dummy_codes)

        torch.cuda.synchronize(device)

        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype)
                logger.info("  Captured CUDA Graph for size=%d", size)
            except Exception:
                logger.warning("  Failed to capture CUDA Graph for size=%d", size, exc_info=True)

        self._warmed_up = True
        logger.info("CUDA Graph warmup complete. Captured %d graphs.", len(self.graphs))

    def _capture_graph_for_size(self, size: int, device: torch.device, dtype: torch.dtype):
        static_input = torch.zeros(
            1,
            self.num_quantizers,
            size,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            _ = self.decoder(static_input)

        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_output = self.decoder(static_input)

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        if codes.shape[0] != 1:
            return self.decoder(codes)

        actual_size = codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        if padded_size is None or padded_size not in self.graphs:
            return self.decoder(codes)

        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:, :, :actual_size] = codes

        self.graphs[padded_size].replay()

        output = self.static_outputs[padded_size]
        total_upsample = self.decoder.total_upsample
        actual_output_len = actual_size * total_upsample

        return output[..., :actual_output_len].clone()

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.decode(codes_chunk)

            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        return torch.cat(wavs, dim=-1)
