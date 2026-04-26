# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray multi-host bootstrap for vLLM eval on multi-VM TPU jobs.

Marin's vLLM eval pipeline assumes a single host: it shells out to ``vllm
serve`` which uses all locally-visible chips with TP. On multi-host TPU
(e.g. v5litepod-16 = 4 VMs x 4 chips), iris coschedules N sibling tasks,
each running the full pipeline independently — so each VM tries to start
its own vllm, they collide, and most fail.

This module orchestrates the Ray-coordinator pattern documented in
``scripts/ray_multihost_vllm/`` (see GH issue #3792):

  Rank 0   : start Ray head on this VM, register address in iris registry,
             pass ``--distributed-executor-backend ray`` plus
             ``--tensor-parallel-size`` and ``--pipeline-parallel-size`` to
             vllm. vllm uses Ray to span all hosts.
  Rank N>0 : poll iris registry for the head address, ``ray start
             --address=... --block``, sit idle until the parent task is
             killed by coscheduling teardown.

Together with ``tpu-inference`` post-2026-03-21 (which has the upstream
multi-host JAX isolation fixes), this lets a single ``vllm serve`` span all
chips of a multi-VM TPU job.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess

from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.env_resources import TaskResources
from rigging.timing import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

_RAY_HEAD_ENDPOINT = "vllm_ray_head"
_RAY_HEAD_DEFAULT_PORT = 6379


def is_multihost_tpu_job() -> bool:
    """True iff this process is running as one of N>1 sibling iris tasks."""
    job_info = get_job_info()
    return job_info is not None and job_info.num_tasks > 1


def derive_tp_pp(job_info: JobInfo | None = None) -> tuple[int, int]:
    """Return ``(tensor_parallel_size, pipeline_parallel_size)`` for this job.

    TP within a VM, PP across VMs. This matches the canonical config from
    issue #3792's ``ray-multihost-vllm`` branch and is the only one we've
    actually seen succeed end-to-end in tpu-inference 0.18.0 on v5litepod-16:
    cross-host TP (TP > chips_per_vm with PP=1) silently dies with
    ``ray.exceptions.ActorDiedError`` during weight load / compile.

    The two PP-related bugs in 0.18.0 are patched at runtime by
    ``marin.inference.tpu_inference_patches`` (anchored to the actual 0.18.0
    source structure).
    """
    info = job_info or get_job_info()
    if info is None:
        raise RuntimeError("derive_tp_pp() requires an iris job context")
    chips_per_vm = TaskResources.from_environment().tpu_count
    if chips_per_vm <= 0:
        raise RuntimeError("TaskResources.tpu_count is 0 — derive_tp_pp() is only valid on TPU jobs")
    return chips_per_vm, info.num_tasks


def start_ray_head(port: int = _RAY_HEAD_DEFAULT_PORT) -> str:
    """Start a Ray head on this VM and advertise it via the iris registry.

    Returns the head address as ``host:port``. Registers an ``atexit`` hook
    that runs ``ray stop`` on process termination.
    """
    info = get_job_info()
    if info is None:
        raise RuntimeError("start_ray_head() requires an iris job context")

    address = f"{info.advertise_host}:{port}"
    # Don't pass --num-cpus 0; Ray must auto-discover host CPU + TPU resources
    # so vllm-tpu's RayDistributedExecutor can place workers on each node. The
    # canonical multi-host vllm-tpu launch script does the same.
    cmd = [
        "ray",
        "start",
        "--head",
        "--port",
        str(port),
        "--node-ip-address",
        info.advertise_host,
        "--disable-usage-stats",
    ]
    logger.info("Starting Ray head: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`ray start --head` failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    ctx = iris_ctx()
    endpoint_id = ctx.registry.register(_RAY_HEAD_ENDPOINT, address)
    atexit.register(_safe_unregister, ctx, endpoint_id)
    atexit.register(stop_ray)
    logger.info("Ray head ready at %s; advertised endpoint %r", address, _RAY_HEAD_ENDPOINT)
    return address


def join_ray_worker_and_block(*, poll_timeout: float = 600.0, poll_interval: float = 2.0) -> None:
    """Resolve the Ray head address and ``ray start --block`` as a worker.

    Replaces the current Python process with ``ray`` via ``execvp`` so that
    iris signals reach Ray directly. Returns only on abnormal exit; in normal
    teardown, iris coscheduling kills this task once rank-0 finishes.

    Applies tpu-inference source-file patches on this VM before joining Ray:
    each iris sibling task has its own ``/app/.venv``, so vllm-serve workers
    spawned on this host (via Ray) only see the patched ``tpu_inference``
    source if we've rewritten it locally first.
    """
    # Importing here to avoid a circular import at module load time.
    from marin.inference.tpu_inference_patches import apply_all as apply_tpu_inference_patches

    apply_tpu_inference_patches()

    ctx = iris_ctx()
    address = _poll_for_ray_head(ctx.resolver, poll_timeout, poll_interval)
    cmd = [
        "ray",
        "start",
        "--address",
        address,
        "--disable-usage-stats",
        "--block",
    ]
    logger.info("Joining Ray cluster as worker: %s", " ".join(cmd))
    os.execvp(cmd[0], cmd)


def stop_ray() -> None:
    """Best-effort ``ray stop --force``. Safe to call multiple times."""
    try:
        subprocess.run(
            ["ray", "stop", "--force"],
            check=False,
            timeout=30,
            capture_output=True,
        )
    except Exception:
        logger.exception("ray stop failed (non-fatal)")


def _safe_unregister(ctx, endpoint_id) -> None:
    try:
        ctx.registry.unregister(endpoint_id)
    except Exception:
        logger.exception("registry.unregister failed (non-fatal)")


def _poll_for_ray_head(resolver, timeout: float, poll_interval: float) -> str:
    result: list[str] = []

    def _check() -> bool:
        resolved = resolver.resolve(_RAY_HEAD_ENDPOINT)
        if not resolved.is_empty:
            result.append(resolved.first().url)
            return True
        return False

    backoff = ExponentialBackoff(initial=poll_interval, maximum=max(poll_interval, 30.0))
    backoff.wait_until_or_raise(
        _check,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Timed out after {timeout}s waiting for Ray head endpoint '{_RAY_HEAD_ENDPOINT}'",
    )
    return result[0]
