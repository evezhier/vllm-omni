import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

logger = init_logger(__name__)


class TalkerMTPCudaGraphWrapper:
    """
    CUDA Graph wrapper for talker_mtp (multi-token prediction).

    Captures the entire MTP pipeline for each batch-size bucket:
    - Code predictor forward
    - Embedding summation
    - Text step addition

    At inference time the wrapper selects the smallest captured bucket that fits
    the actual batch size, zero-pads the inputs, replays the corresponding graph,
    and returns only the non-padded output rows.
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
        max_batch_size: int = 1,
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

        self.batch_sizes = self._compute_bucket_sizes(max_batch_size)

        # Per-bucket static GPU buffers, keyed by batch size.
        self._buffers: dict[int, dict[str, torch.Tensor]] = {}
        for bs in self.batch_sizes:
            self._buffers[bs] = {
                "input_ids": torch.zeros(bs, 1, dtype=torch.long, device=device),
                "last_id_hidden": torch.zeros(bs, 1, self.hidden_size, dtype=torch.bfloat16, device=device),
                "past_hidden": torch.zeros(bs, 1, self.hidden_size, dtype=torch.bfloat16, device=device),
                "text_step": torch.zeros(bs, 1, self.hidden_size, dtype=torch.bfloat16, device=device),
                "audio_codes": torch.zeros(bs, self.num_code_groups, dtype=torch.long, device=device),
                "inputs_embeds": torch.zeros(bs, self.hidden_size, dtype=torch.bfloat16, device=device),
            }

        # Current bucket's buffer dict; always set before _mtp_forward() is called.
        self._active_bufs: dict[str, torch.Tensor] = self._buffers[self.batch_sizes[0]]

        self.graphs: dict[int, CUDAGraph] = {}

        self.num_warmup_steps = num_warmup_steps
        self.warmed_up = False
        self.captured = False

    def _compute_bucket_sizes(self, max_batch_size: int) -> list[int]:
        """Return sorted list of CUDA-graph bucket sizes covering 1..max_batch_size.

        Uses powers of 2 up to max_batch_size, then appends max_batch_size itself
        if it is not already a power of 2. Always includes 1.
        """
        sizes: list[int] = []
        b = 1
        while b <= max_batch_size:
            sizes.append(b)
            b *= 2
        if sizes[-1] < max_batch_size:
            sizes.append(max_batch_size)
        return sizes

    @property
    def input_ids_buf(self) -> torch.Tensor:
        return self._active_bufs["input_ids"]

    @property
    def last_id_hidden_buf(self) -> torch.Tensor:
        return self._active_bufs["last_id_hidden"]

    @property
    def past_hidden_buf(self) -> torch.Tensor:
        return self._active_bufs["past_hidden"]

    @property
    def text_step_buf(self) -> torch.Tensor:
        return self._active_bufs["text_step"]

    @property
    def audio_codes_buf(self) -> torch.Tensor:
        return self._active_bufs["audio_codes"]

    @property
    def inputs_embeds_out_buf(self) -> torch.Tensor:
        return self._active_bufs["inputs_embeds"]

    @property
    def graph(self) -> CUDAGraph | None:
        return self.graphs.get(1)

    @torch.inference_mode
    def _mtp_forward(self):
        """Run the full MTP pipeline once; this is the function captured by the graph.

        Calls the code predictor to generate residual codebook tokens, then
        accumulates their embeddings together with the layer-0 hidden state and
        the text step to produce the next-step input embedding.
        Results are written into the active output buffers (_active_bufs).
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

        bs = self.input_ids_buf.shape[0]
        summed = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        result = (summed + self.text_step_buf).reshape(bs, -1)
        self.inputs_embeds_out_buf.copy_(result)

    def capture(self):
        """Warm up and capture _mtp_forward as CUDA graphs for every bucket size.
        Running the largest batch size first ensures that the code predictor's
        _proj_buf is sized for the max batch size and is not reallocated.
        """
        for bs in reversed(self.batch_sizes):
            self._active_bufs = self._buffers[bs]
            for _ in range(self.num_warmup_steps):
                self._mtp_forward()
        torch.cuda.synchronize(self.device)

        for bs in self.batch_sizes:
            self._active_bufs = self._buffers[bs]

            # Capture on a dedicated side stream so the default stream is not
            # polluted by graph memory.
            with torch.cuda.device(self.device_index):
                graph = CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    # One additional warmup on the capture stream.
                    self._mtp_forward()
                    s.synchronize()
                    with torch.cuda.graph(graph):
                        self._mtp_forward()

            torch.cuda.current_stream().wait_stream(s)
            self.graphs[bs] = graph

        torch.cuda.synchronize()
        self.captured = True

    def warmup(self, device: torch.device):
        """Capture CUDA graphs for all batch-size buckets on the given device."""
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
        logger.info(
            "TalkerMTPCudaGraphWrapper: CUDA graphs captured for batch sizes %s",
            self.batch_sizes,
        )

    @torch.inference_mode()
    def _talker_mtp(self, input_ids, last_id_hidden, past_hidden, text_step):
        """Run one MTP step via graph replay.
        Zero-pads the inputs to the smallest fitting bucket, unpads after replay.

        Args:
            input_ids:      Layer-0 token ids,            shape [B] or [B, 1].
            last_id_hidden: Layer-0 hidden state,         shape [B, H] or [B, 1, H].
            past_hidden:    Previous talker hidden state, shape [B, H] or [B, 1, H].
            text_step:      Current text hidden state,    shape [B, H] or [B, 1, H].

        Returns:
            (inputs_embeds, audio_codes): shapes [B, H] and [B, num_code_groups].

        Raises:
            RuntimeError: If warmup() has not been called yet.
            ValueError:   If B exceeds the maximum captured bucket size.
        """
        if not self.captured or not self.graphs:
            raise RuntimeError("TalkerMTPCudaGraphWrapper: graph not captured — call warmup() first")

        actual_bs = input_ids.shape[0]
        target_bs = min((b for b in self.graphs if b >= actual_bs), default=None)
        if target_bs is None:
            logger.warning(
                "TalkerMTPCudaGraphWrapper: batch size %d exceeds max captured bucket %d, "
                "falling back to eager execution",
                actual_bs,
                max(self.graphs),
            )
            return self.talker._talker_mtp(input_ids, last_id_hidden, past_hidden, text_step)

        bufs = self._buffers[target_bs]

        bufs["input_ids"][:actual_bs].copy_(input_ids.reshape(actual_bs, 1))
        bufs["last_id_hidden"][:actual_bs].copy_(last_id_hidden.reshape(actual_bs, 1, -1))
        bufs["past_hidden"][:actual_bs].copy_(past_hidden.reshape(actual_bs, 1, -1))
        bufs["text_step"][:actual_bs].copy_(text_step.reshape(actual_bs, 1, -1))

        if actual_bs < target_bs:
            bufs["input_ids"][actual_bs:].zero_()
            bufs["last_id_hidden"][actual_bs:].zero_()
            bufs["past_hidden"][actual_bs:].zero_()
            bufs["text_step"][actual_bs:].zero_()

        self.graphs[target_bs].replay()

        return bufs["inputs_embeds"][:actual_bs].clone(), bufs["audio_codes"][:actual_bs].clone()
