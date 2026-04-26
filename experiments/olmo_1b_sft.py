# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Fine-tune OLMo-2 1B on GSM8K and evaluate on GSM8K, swept over a set of
stage-1 pretraining checkpoints.

Per OLMo 2 1B SFT spec comment: GSM8K is *eval-only* in the OLMo paper, so
this is not the official OLMo procedure — it's a per-checkpoint "can this
base model learn to do grade-school math?" probe.

For each revision we build THREE steps:
  1. BASELINE eval: the raw HF checkpoint evaluated on gsm8k_cot 8-shot
     with `apply_chat_template=False` (base models don't understand chat
     tags — turning the template on would corrupt their few-shot pattern
     completion). This measures the pre-FT math ability.
  2. SFT step: chat-SFT on GSM8K train (7,473 examples, 3 epochs, lr 2e-5,
     bs 32), loading HF weights from allenai/OLMo-2-0425-1B at that revision.
  3. POST-FT eval: the SFT'd model on gsm8k_cot 8-shot with
     `apply_chat_template=True` (the chat template is now the canonical
     prompt format for the FT'd model).

The lift (post-FT - baseline) at each revision shows how much of the math
signal is latent in pretraining vs. unlocked by FT.
"""

import math

from transformers import AutoConfig

from fray.cluster import ResourceConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness, extract_model_name_and_path
from experiments.models import ModelConfig, download_model_step
from experiments.posttrain.instruction_datasets import (
    InstructionDatasetConfig,
    instruction_response_adapter,
    transform_dataset_step,
)
from experiments.simple_sft_config import SimpleSFTConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.models.olmo import Olmo2Config
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

OLMO2_1B_REPO = "allenai/OLMo-2-0425-1B"

# Pull the tokenizer from the SFT release: base's tokenizer has no chat_template,
# while SFT's does (<|endoftext|><|user|>…<|assistant|>…<|endoftext|>). Vocabs
# are byte-identical between base and SFT (vocab_size 100278, same special tokens).
OLMO2_TOKENIZER = "allenai/OLMo-2-0425-1B-SFT"
OLMO2_EOS_TOKEN_ID = 100257  # <|endoftext|>

# OLMo-2's published chat template doesn't include the {% generation %} markers
# that Levanter's ChatProcessor needs to build the assistant-mask (so loss is
# computed only on assistant tokens). This is the same template, with markers
# wrapping the assistant content + EOS. Verified via apply_chat_template that
# the rendered string is identical and that the resulting assistant_mask covers
# exactly the answer tokens and the trailing <|endoftext|>.
OLMO2_TRAINABLE_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' }}"
    "{%- generation %}"
    "{% if not loop.last %}{{ message['content'] + eos_token + '\n' }}"
    "{% else %}{{ message['content'] + eos_token }}{% endif %}"
    "{%- endgeneration %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    "{% endfor %}"
)

# Model dims are identical across all OLMo-2-0425-1B revisions; read once
# from the base repo and reuse.
_olmo2_1b_hf_config = AutoConfig.from_pretrained(OLMO2_1B_REPO, trust_remote_code=True)
olmo2_1b_config = Olmo2Config.from_hf_config(_olmo2_1b_hf_config)

# GSM8K 'main' train split: 7,473 {question, answer} pairs.
# We don't register GSM8K in `INSTRUCTION_DATASET_NAME_TO_CONFIG` because
# the catalog is meant for reusable instruction-tuning mixtures (Tulu,
# SmolTalk, Nemotron-v2). GSM8K is a single-turn math benchmark used here
# as a one-off FT corpus — keeping its config local makes the experiment
# self-contained and keeps the shared catalog focused.
#
# `transform_dataset_step` downloads the HF dataset and runs the adapter
# to reshape rows into [{role:user, content:question},
# {role:assistant, content:answer}] in the OpenAI messages format.
# `ChatLmDatasetFormat()` then applies the tokenizer's chat template and
# masks the user turn so loss is only on the answer.
_gsm8k_config = InstructionDatasetConfig(
    hf_dataset_id="openai/gsm8k",
    revision="740312add88f781978c0658806c59bc2815b9866",
    adapter=instruction_response_adapter(
        instruction_column="question",
        response_column="answer",
    ),
    metadata_columns=[],
    name="openai/gsm8k",
    subsets=["main"],
    splits=["train"],
)
gsm8k_dataset = transform_dataset_step(_gsm8k_config)

gsm8k_tokenized = default_tokenize(
    name="gsm8k_main_train_olmo2_tokenizer",
    dataset=gsm8k_dataset / "**/*.jsonl.gz",
    tokenizer=OLMO2_TOKENIZER,
    format=ChatLmDatasetFormat(chat_template=OLMO2_TRAINABLE_CHAT_TEMPLATE),
)

# ---------------------------------------------------------------------------
# Hyperparameters (shared across the sweep)
# ---------------------------------------------------------------------------
# GSM8K-FT recipe: batch 32, lr 2e-5, 3 epochs, linear warmup+decay.
# 7473 * 3 / 32 = 700.6 → 701 steps; small enough that we want a short warmup.
NUM_TRAIN_EXAMPLES = 7_473
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
NUM_TRAIN_STEPS = math.ceil(NUM_EPOCHS * NUM_TRAIN_EXAMPLES / TRAIN_BATCH_SIZE)  # 701

TRAIN_SEQ_LEN = 1024  # GSM8K: questions ~200 tok, answers ~400 tok; 1024 is ample.

# Just the fully-cooked post-midtraining base model (stage1 + stage2).
# `main` on allenai/OLMo-2-0425-1B = base release (not -SFT / -Instruct).
# Extend this list to fan out across stage-1 intermediate revisions.
REVISIONS: list[str] = [
    "main",
]


def sft_config_for(revision: str) -> SimpleSFTConfig:
    return SimpleSFTConfig(
        resources=ResourceConfig.with_tpu("v5litepod-16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=2e-5,
        tokenizer=OLMO2_TOKENIZER,
        initialize_from_hf=f"{OLMO2_1B_REPO}@{revision}",
        max_seq_len=TRAIN_SEQ_LEN,
        lr_schedule="linear",
        # 5% warmup then linear decay to min_lr over the remaining 95%.
        # SimpleSFTConfig.decay defaults to 0.0 (no decay phase) — set explicitly.
        warmup=0.05,
        decay=0.95,
        min_lr_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=0,
        steps_per_hf_export=max(NUM_TRAIN_STEPS // 2, 200),
        hf_generation_eos_token_ids=[OLMO2_EOS_TOKEN_ID],
    )


def _vllm_gsm8k_eval(step, *, apply_chat_template: bool, resource_config: ResourceConfig) -> ExecutorStep:
    """vLLM lm-eval-harness GSM8K 8-shot CoT eval on multi-host TPU."""
    name, model_path = extract_model_name_and_path(step)
    return evaluate_lm_evaluation_harness(
        model_name=name,
        model_path=model_path,
        evals=[EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot")],
        resource_config=resource_config,
        apply_chat_template=apply_chat_template,
    )


def build_sweep_steps(revisions: list[str]) -> list[ExecutorStep]:
    """For each revision: (download, base eval, SFT, post-FT eval)."""
    steps: list[ExecutorStep] = []
    eval_resources = ResourceConfig.with_tpu("v5litepod-16")
    for revision in revisions:
        slug = revision.replace("/", "_").replace("-", "_")

        # 1. Baseline: eval the raw checkpoint with NO chat template.
        base_checkpoint = download_model_step(
            ModelConfig(hf_repo_id=OLMO2_1B_REPO, hf_revision=revision),
        )
        base_eval = _vllm_gsm8k_eval(base_checkpoint, apply_chat_template=False, resource_config=eval_resources)

        # 2. SFT on GSM8K train.
        sft_step = default_sft(
            name=f"olmo2_1b_gsm8k_ft/{slug}",
            tokenized=gsm8k_tokenized,
            model_config=olmo2_1b_config,  # type: ignore[arg-type]  # LlamaConfig hint is narrow; runtime accepts LmConfig
            sft_config=sft_config_for(revision),
            tags=["olmo2", "1b", "sft", "gsm8k", revision],
        )

        # 3. Post-FT: eval the SFT'd model WITH chat template.
        post_ft_eval = _vllm_gsm8k_eval(sft_step, apply_chat_template=True, resource_config=eval_resources)
        del sft_step, post_ft_eval  # TODO: re-include after base eval validates pipeline against paper's 43.8
        steps.extend([base_checkpoint, base_eval])
    return steps


if __name__ == "__main__":
    executor_main(
        steps=build_sweep_steps(REVISIONS),
        description="OLMo 2 1B — GSM8K FT + eval across stage-1 checkpoints",
    )
