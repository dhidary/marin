# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime source-file patches for tpu-inference 0.18.0 multi-host PP bugs.

Ports the two patches from the ``ray-multihost-vllm`` branch
(``scripts/ray_multihost_vllm/patches/``) that the issue's 2026-03-29 update
flagged as "still needed, not yet upstream":

  - ``patch_pp_parallel_state.py``  → ``tpu_worker.py``: when running on Ray
    multi-host TPU, vLLM initializes its PP group with ``world_size=1, rank=0``
    on every worker, so ``make_layers`` -> ``get_pp_indices`` returns the
    rank-0 layer slice for ALL workers. We override the PP group's attrs with
    the worker's actual TPU PP rank.

  - ``patch_kv_cache_local_names.py`` → ``kv_cache_manager.py``: KV caches on
    non-rank-0 PP stages get registered under rank-0 layer names (e.g.
    ``model.layers.0..3``) but the model on stage 1 references its own local
    layer names (``model.layers.4..7``), so attention forward fails with
    ``KeyError: 'model.layers.4.self_attn.attn'``. We re-register the KV cache
    index using the worker's *local* attention layer names and allocate extras
    if the count doesn't match.

Applied at vllm-serve startup by mutating files inside the installed
``tpu_inference`` package. Idempotent — patches detect their own marker and
skip if already applied. The vLLM serve subprocess (and its Ray worker actors
on other hosts) re-import the patched files cleanly.
"""

from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_PP_GROUP_MARKER = "PP parallel state override"
_KV_CACHE_MARKER = "PP KV cache fix"


def _tpu_inference_dir() -> str | None:
    spec = importlib.util.find_spec("tpu_inference")
    if spec is None or spec.origin is None:
        return None
    return os.path.dirname(spec.origin)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def _patch_tpu_worker_pp_rank(tpu_inference_dir: str) -> None:
    path = os.path.join(tpu_inference_dir, "worker", "tpu_worker.py")
    if not os.path.exists(path):
        logger.warning("tpu_inference patch: %s not found, skipping pp_rank patch", path)
        return
    code = _read(path)
    if _PP_GROUP_MARKER in code:
        logger.info("tpu_inference patch: pp_rank already applied to %s", path)
        return

    target = (
        "            ensure_model_parallel_initialized(\n"
        "                tensor_model_parallel_size=1,\n"
        "                pipeline_model_parallel_size=1,\n"
        "            )"
    )
    if target not in code:
        logger.warning("tpu_inference patch: pp_rank target not found in %s", path)
        return

    replacement = (
        target
        + "\n        # Override the vLLM PP group to match the actual TPU PP rank.\n"
        + "        # vLLM was initialized with world_size=1,rank=0 (single chip),\n"
        + "        # but the vLLM model's make_layers needs the real PP rank to\n"
        + "        # assign the correct layers to each worker.\n"
        + "        pp_size = self.parallel_config.pipeline_parallel_size\n"
        + "        if pp_size > 1:\n"
        + "            from vllm.distributed.parallel_state import get_pp_group\n"
        + "            pp_group = get_pp_group()\n"
        + "            pp_group.rank = self.rank\n"
        + "            pp_group.ranks = list(range(pp_size))\n"
        + "            pp_group.rank_in_group = self.rank\n"
        + "            pp_group.world_size = pp_size\n"
        + f"            logger.info(  # {_PP_GROUP_MARKER}\n"
        + '                f"PP parallel state override: rank={self.rank}, "\n'
        + '                f"world_size={pp_size}, ranks={pp_group.ranks}")'
    )
    _write(path, code.replace(target, replacement))
    logger.info("tpu_inference patch: applied pp_rank override to %s", path)


def _patch_kv_cache_local_names(tpu_inference_dir: str) -> None:
    path = os.path.join(tpu_inference_dir, "runner", "kv_cache_manager.py")
    if not os.path.exists(path):
        logger.warning("tpu_inference patch: %s not found, skipping kv_cache patch", path)
        return
    code = _read(path)
    if _KV_CACHE_MARKER in code:
        logger.info("tpu_inference patch: kv_cache already applied to %s", path)
        return

    # Anchor on the new 0.18.0 log construction (the older `f"Init kv-cache | "
    # f"num_layers=..."` shape doesn't exist in 0.18.0). Inject the fix right
    # before this line; at this point in `initialize_kv_cache`, the variables
    # `kv_caches`, `self.shared_kv_cache_layers`, `self.runner.layer_name_to_kvcache_index`,
    # and `self.runner.vllm_config` are all in scope.
    target = '        log_parts = [\n            "Init kv-cache", f"num_total_layers={len(kv_caches)}",'
    if target not in code:
        logger.warning("tpu_inference patch: kv_cache target not found in %s", path)
        return

    # Re-registration only — no extra-cache allocation. OLMo-2-1B has 16 layers
    # / PP=4 = 4 per stage (divisible), so cache-count == local-layer-count
    # always holds. The branch's "allocate extras" path was specific to
    # non-divisible counts and needs `head_size` which isn't in this scope.
    fix = (
        "        # Fix for PP (issue #3792 patch #6, adapted to tpu-inference 0.18.0):\n"
        "        # re-register KV cache index with the LOCAL attention layer names so\n"
        "        # PP-stage workers (rank > 0) can look up KV caches by their own layer\n"
        "        # names instead of rank-0's. Without this, stage>0 workers hit\n"
        "        # KeyError: 'model.layers.N.self_attn.attn' during compile.\n"
        "        try:\n"
        "            from vllm.model_executor.layers.attention import Attention\n"
        "            from vllm.config import get_layers_from_vllm_config\n"
        "            _local_layers = get_layers_from_vllm_config(\n"
        "                self.runner.vllm_config, Attention)\n"
        "            for _shared in self.shared_kv_cache_layers:\n"
        "                _local_layers.pop(_shared, None)\n"
        "\n"
        "            def _layer_sort_key(n):\n"
        "                if 'layers.' in n:\n"
        "                    _digits = ''.join(c for c in n.split('layers.')[1].split('.')[0] if c.isdigit())\n"
        "                    return int(_digits) if _digits else 0\n"
        "                return 0\n"
        "\n"
        "            _local_names = sorted(_local_layers.keys(), key=_layer_sort_key)\n"
        "            if _local_names and len(_local_names) <= len(kv_caches):\n"
        "                self.runner.layer_name_to_kvcache_index.clear()\n"
        "                for _i, _n in enumerate(_local_names):\n"
        "                    self.runner.layer_name_to_kvcache_index[_n] = _i\n"
        "                logger.info(\n"
        f"                    \"{_KV_CACHE_MARKER}: registered %d local names (caches=%d): %s..%s\",\n"
        "                    len(_local_names), len(kv_caches),\n"
        "                    _local_names[0], _local_names[-1])\n"
        "        except Exception as _e:\n"
        f"            logger.warning(\"{_KV_CACHE_MARKER} failed: %s\", _e)\n"
        "\n"
    )
    _write(path, code.replace(target, fix + target))
    logger.info("tpu_inference patch: applied kv_cache local-names fix to %s", path)


def apply_all() -> None:
    """Apply all tpu-inference multi-host PP patches. Idempotent."""
    tpu_inference_dir = _tpu_inference_dir()
    if tpu_inference_dir is None:
        logger.info("tpu_inference patches: package not installed, nothing to patch")
        return
    _patch_tpu_worker_pp_rank(tpu_inference_dir)
    _patch_kv_cache_local_names(tpu_inference_dir)
