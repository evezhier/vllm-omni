import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

logger = init_logger(__name__)


class TalkerMTPCudaGraphWrapper:
    """
    CUDA Graph wrapper for talker_mtp (multi-token prediction).

    Captures the entire MTP pipeline:
    - Code predictor forward
    - Embedding summation
    - Text step addition
    """

    def __init__(
        self,
        talker_model,
        talker_config,
        device="cuda",
        enabled=True,
        temperature=0.9,
        top_k=50,
        num_warmup_steps=3,
    ):
        self.device = device
        self.device_index = torch.device(device).index or 0
        self.enabled = enabled

        self.talker = talker_model
        self.code_predictor = talker_model.code_predictor
        self.num_code_groups = talker_config.num_code_groups
        self.hidden_size = talker_config.hidden_size
        self.vocab_size = talker_model.code_predictor.config.vocab_size
        self.temperature = temperature
        self.top_k = top_k

        # Static input buffers (fixed on GPU)
        self.input_ids_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.last_id_hidden_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.bfloat16, device=device)
        self.past_hidden_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.bfloat16, device=device)
        self.text_step_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.bfloat16, device=device)

        # Static output buffers
        self.audio_codes_buf = torch.zeros(1, self.num_code_groups, dtype=torch.long, device=device)
        self.inputs_embeds_out_buf = torch.zeros(1, self.hidden_size, dtype=torch.bfloat16, device=device)

        self.num_warmup_steps = num_warmup_steps
        self.warmed_up = False
        self.graph = None
        self.captured = False

    @torch.inference_mode
    def _mtp_forward(self):
        """Run the full MTP pipeline once; this is the function captured by the graph.

        Calls the code predictor to generate residual codebook tokens, then
        accumulates their embeddings together with the layer-0 hidden state and
        the text step to produce the next-step input embedding.
        Results are written into the static output buffers.
        """
        audio_codes = self.code_predictor.forward(
            layer0_code=self.input_ids_buf,
            layer0_embed=self.last_id_hidden_buf,
            last_talker_hidden=self.past_hidden_buf,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        self.audio_codes_buf.copy_(audio_codes)

        layer0 = self.audio_codes_buf[:, :1]
        invalid0 = (layer0 < 0) | (layer0 >= int(self.vocab_size))
        self.audio_codes_buf.masked_fill_(invalid0.expand_as(self.audio_codes_buf), 0)
        residual_ids = self.audio_codes_buf[:, 1:]

        embeds = [self.last_id_hidden_buf]
        for i in range(self.num_code_groups - 1):
            emb = self.code_predictor.get_input_embeddings()[i](residual_ids[:, i : i + 1])
            embeds.append(emb)

        summed = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        result = (summed + self.text_step_buf).reshape(1, -1)
        self.inputs_embeds_out_buf.copy_(result)

    def capture(self):
        """Warm up and capture _mtp_forward as a CUDA graph, pre-allocate CUDA memory."""
        for _ in range(self.num_warmup_steps):
            self._mtp_forward()
        torch.cuda.synchronize(self.device)

        # Capture on a dedicated side stream so the default stream is not
        # polluted by graph memory.
        with torch.cuda.device(self.device_index):
            self.graph = CUDAGraph()
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

    def warmup(self, device: torch.device):
        """Capture the CUDA graph on the given device."""
        if not self.enabled:
            logger.info("TalkerMTPCudaGraphWrapper: disabled, skipping capture")
            return
        if device.type != "cuda":
            logger.info("CUDA Graph warmup skipped: device %s is not CUDA", device)
            return
        if self.warmed_up:
            logger.warning("CUDA Graph already warmed up, skipping")
            return
        self.device = device
        self.device_index = device.index or 0
        self.capture()
        self.warmed_up = True
        logger.info("TalkerMTPCudaGraphWrapper: CUDA graph captured")

    @torch.inference_mode()
    def _talker_mtp(self, input_ids, last_id_hidden, past_hidden, text_step):
        """Run one MTP step via graph replay.

        Copies the four input tensors into the static GPU buffers, replays the
        captured graph, and returns clones of the output buffers so callers
        receive independent tensors.
        Currently supports bs=1 only.

        Args:
            input_ids: Layer-0 token id, shape broadcastable to [1, 1].
            last_id_hidden: Layer-0 hidden state, shape broadcastable to [1, 1, H].
            past_hidden: Previous talker hidden state, shape broadcastable to [1, 1, H].
            text_step: Current text decoder hidden state, shape broadcastable to [1, 1, H].

        Returns:
            (inputs_embeds, audio_codes): shapes [1, H] and [1, num_code_groups].

        Raises:
            RuntimeError: If warmup() has not been called yet.
        """
        if not self.captured or self.graph is None:
            raise RuntimeError("TalkerMTPCudaGraphWrapper: graph not captured — call warmup() first")

        if input_ids.shape[0] != 1:
            raise ValueError(f"TalkerMTPCudaGraphWrapper only supports bs=1, got input_ids.shape={input_ids.shape}")

        self.input_ids_buf.copy_(input_ids.reshape(1, 1))
        self.last_id_hidden_buf.copy_(last_id_hidden.reshape(1, 1, -1))
        self.past_hidden_buf.copy_(past_hidden.reshape(1, 1, -1))
        self.text_step_buf.copy_(text_step.reshape(1, 1, -1))

        self.graph.replay()

        return self.inputs_embeds_out_buf.clone(), self.audio_codes_buf.clone()
