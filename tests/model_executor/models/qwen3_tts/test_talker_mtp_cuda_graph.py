# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for TalkerMTPCudaGraphWrapper.

Verifies:
  - Warmup / graph capture mechanics.
  - Output shape and validity of audio codes.
  - Numerical equivalence with sampling disabled (no randomness).
  - Batch size > 1 support via bucket-based graph capture.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

DEVICE = torch.device("cuda:0")
VOCAB_SIZE = 8
NUM_CODE_GROUPS = 3
HIDDEN_SIZE = 16

# ---------------------------------------------------------------------------
# Module import (package or direct file path fallback)
# ---------------------------------------------------------------------------

try:
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_talker_wrapper import (
        TalkerMTPCudaGraphWrapper,
    )
except Exception:
    _WRAPPER_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            os.pardir,
            os.pardir,
            "vllm_omni",
            "model_executor",
            "models",
            "qwen3_tts",
            "cuda_graph_decoder_wrapper.py",
        )
    )
    _spec = importlib.util.spec_from_file_location("cuda_graph_decoder_wrapper", _WRAPPER_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    TalkerMTPCudaGraphWrapper = _mod.TalkerMTPCudaGraphWrapper


# ---------------------------------------------------------------------------
# Synthetic models that mimic the real talker / code-predictor interface
# ---------------------------------------------------------------------------


class SyntheticCodePredictorConfig:
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        num_code_groups: int = NUM_CODE_GROUPS,
        hidden_size: int = HIDDEN_SIZE,
    ):
        self.vocab_size = vocab_size
        self.num_code_groups = num_code_groups
        self.hidden_size = hidden_size


class SyntheticTalkerConfig:
    def __init__(
        self,
        num_code_groups: int = NUM_CODE_GROUPS,
        hidden_size: int = HIDDEN_SIZE,
    ):
        self.num_code_groups = num_code_groups
        self.hidden_size = hidden_size


class SyntheticCodePredictor(nn.Module):
    def __init__(self, config: SyntheticCodePredictorConfig):
        super().__init__()
        self.config = config
        self._num_groups = config.num_code_groups

        # One lm_head per residual step (steps 1 .. Q-1)
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        # Codec embeddings for embedding-summation step inside the wrapper
        self._codec_embeddings = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.hidden_size) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self._codec_embeddings

    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        bsz = int(layer0_code.shape[0])
        device = layer0_code.device

        all_codes = torch.zeros(bsz, self._num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0

        # Use last_talker_hidden as the shared hidden state for all steps
        hidden = last_talker_hidden.reshape(bsz, -1).to(self.lm_heads[0].weight.dtype)

        for step in range(1, self._num_groups):
            logits = self.lm_heads[step - 1](hidden)  # [bsz, vocab_size]

            if use_sampling:
                scaled = logits * inv_temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)

            all_codes[:, step] = next_ids.reshape(bsz)

        return all_codes


class SyntheticTalkerModel:
    def __init__(self, predictor_config: SyntheticCodePredictorConfig):
        self.code_predictor = SyntheticCodePredictor(predictor_config).to(device=DEVICE, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predictor_config():
    return SyntheticCodePredictorConfig()


@pytest.fixture(scope="module")
def talker_config():
    return SyntheticTalkerConfig()


@pytest.fixture(scope="module")
def talker_model(predictor_config):
    torch.manual_seed(0)
    return SyntheticTalkerModel(predictor_config)


@pytest.fixture(scope="module")
def wrapper(talker_model, talker_config):
    w = TalkerMTPCudaGraphWrapper(
        talker_model=talker_model,
        talker_config=talker_config,
        device=DEVICE,
        enabled=True,
        temperature=0.9,
        top_k=VOCAB_SIZE,  # allow all tokens
        max_batch_size=4,
    )
    w.warmup(DEVICE)
    return w


def _random_inputs(bs: int = 1, hidden_size: int = HIDDEN_SIZE, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB_SIZE, (bs,), dtype=torch.long, device=DEVICE)
    last_id_hidden = torch.randn(bs, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    past_hidden = torch.randn(bs, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    text_step = torch.randn(bs, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    return input_ids, last_id_hidden, past_hidden, text_step


# ---------------------------------------------------------------------------
# 1. Warmup / capture mechanics and output shapes
# ---------------------------------------------------------------------------


def test_warmup_sets_captured_flag(talker_model, talker_config):
    w = TalkerMTPCudaGraphWrapper(
        talker_model=talker_model,
        talker_config=talker_config,
        device=DEVICE,
        enabled=True,
        top_k=VOCAB_SIZE,
    )
    assert not w.captured
    assert w.graph is None
    w.warmup(DEVICE)
    assert w.captured
    assert w.graph is not None


def test_captures_all_buckets(talker_model, talker_config):
    """Wrapper should capture one graph per bucket size."""
    w = TalkerMTPCudaGraphWrapper(
        talker_model=talker_model,
        talker_config=talker_config,
        device=DEVICE,
        enabled=True,
        top_k=VOCAB_SIZE,
        max_batch_size=4,
    )
    w.warmup(DEVICE)
    assert set(w.graphs.keys()) == {1, 2, 4}


@pytest.mark.parametrize("bs", [1, 2, 3, 4])
def test_output_shapes(wrapper, bs):
    inputs = _random_inputs(bs=bs, seed=42)
    inputs_embeds, audio_codes = wrapper._talker_mtp(*inputs)

    assert inputs_embeds.shape == (bs, HIDDEN_SIZE), (
        f"Expected inputs_embeds shape ({bs}, {HIDDEN_SIZE}), got {inputs_embeds.shape}"
    )
    assert audio_codes.shape == (bs, NUM_CODE_GROUPS), (
        f"Expected audio_codes shape ({bs}, {NUM_CODE_GROUPS}), got {audio_codes.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Numerical equivalence with sampling disabled: CUDA graph vs eager
# ---------------------------------------------------------------------------


class _ArgmaxWrapper(TalkerMTPCudaGraphWrapper):
    """TalkerMTPCudaGraphWrapper variant that captures argmax (do_sample=False)."""

    @torch.inference_mode
    def _mtp_forward(self):
        audio_codes = self.code_predictor.forward(
            layer0_code=self.input_ids_buf,
            layer0_embed=self.last_id_hidden_buf,
            last_talker_hidden=self.past_hidden_buf,
            do_sample=False,
        )
        self.audio_codes_buf.copy_(audio_codes)

        bs = audio_codes.shape[0]
        residual_ids = audio_codes[:, 1:]
        self.inputs_embeds_out_buf.copy_(self.last_id_hidden_buf.reshape(bs, -1))
        codec_embeds = self.code_predictor.get_input_embeddings()
        for i in range(self.num_code_groups - 1):
            self.inputs_embeds_out_buf.add_(codec_embeds[i](residual_ids[:, i : i + 1]).reshape(bs, -1))
        self.inputs_embeds_out_buf.add_(self.text_step_buf.reshape(bs, -1))


def _eager_mtp_argmax(predictor, input_ids, last_id_hidden, past_hidden, text_step):
    bsz = input_ids.shape[0]
    num_groups = predictor.config.num_code_groups
    with torch.inference_mode():
        audio_codes = predictor.forward(
            layer0_code=input_ids.reshape(bsz, 1),
            layer0_embed=last_id_hidden.reshape(bsz, 1, -1),
            last_talker_hidden=past_hidden.reshape(bsz, 1, -1),
            do_sample=False,
        )

        residual_ids = audio_codes[:, 1:]
        inputs_embeds = last_id_hidden.reshape(bsz, -1).clone()
        codec_embeds = predictor.get_input_embeddings()
        for i in range(num_groups - 1):
            inputs_embeds.add_(codec_embeds[i](residual_ids[:, i : i + 1]).reshape(bsz, -1))
        inputs_embeds.add_(text_step.reshape(bsz, -1))

    return inputs_embeds, audio_codes


@pytest.mark.parametrize("seed", [42, 99, 1])
def test_graph_matches_eager_argmax(talker_model, talker_config, seed):
    """Argmax CUDA graph must be bit-identical to eager argmax (bs=1)."""
    w = _ArgmaxWrapper(
        talker_model=talker_model,
        talker_config=talker_config,
        device=DEVICE,
        enabled=True,
        top_k=VOCAB_SIZE,
    )
    w.warmup(DEVICE)

    input_ids, last_id_hidden, past_hidden, text_step = _random_inputs(bs=1, seed=seed)
    graph_embeds, graph_codes = w._talker_mtp(input_ids, last_id_hidden, past_hidden, text_step)
    eager_embeds, eager_codes = _eager_mtp_argmax(
        talker_model.code_predictor, input_ids, last_id_hidden, past_hidden, text_step
    )

    torch.testing.assert_close(
        graph_codes, eager_codes, atol=0, rtol=0, msg="audio_codes mismatch (argmax, no sampling)"
    )
    torch.testing.assert_close(
        graph_embeds, eager_embeds, atol=0, rtol=0, msg="inputs_embeds mismatch (argmax, no sampling)"
    )


@pytest.mark.parametrize("bs", [2, 3, 4])
@pytest.mark.parametrize("seed", [42, 7])
def test_graph_matches_eager_argmax_batched(talker_model, talker_config, bs, seed):
    """Argmax CUDA graph must be bit-identical to eager argmax for bs > 1."""
    w = _ArgmaxWrapper(
        talker_model=talker_model,
        talker_config=talker_config,
        device=DEVICE,
        enabled=True,
        top_k=VOCAB_SIZE,
        max_batch_size=4,
    )
    w.warmup(DEVICE)

    input_ids, last_id_hidden, past_hidden, text_step = _random_inputs(bs=bs, seed=seed)
    graph_embeds, graph_codes = w._talker_mtp(input_ids, last_id_hidden, past_hidden, text_step)
    eager_embeds, eager_codes = _eager_mtp_argmax(
        talker_model.code_predictor, input_ids, last_id_hidden, past_hidden, text_step
    )

    torch.testing.assert_close(graph_codes, eager_codes, atol=0, rtol=0, msg=f"audio_codes mismatch (argmax, bs={bs})")
    torch.testing.assert_close(
        graph_embeds, eager_embeds, atol=0, rtol=0, msg=f"inputs_embeds mismatch (argmax, bs={bs})"
    )
