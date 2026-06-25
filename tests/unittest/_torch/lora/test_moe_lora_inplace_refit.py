# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place refit of routed-expert MoE LoRA adapters ("Plan B").

SCOPE: adapter-only. In multilora routed-expert RL the base model is FROZEN —
each step updates only the LoRA adapter A/B (verified against xorl's
multilora_rl_recipe.py). The base-weight refit path (rlhf_utils.update_weights)
is NOT in scope; only its IPC transport (reduce_tensor -> ipc_handles ->
_collective_rpc) is reused, pointed at the PEFT cache instead of model.load_weights.

Two groups of tests:

1. RUNNABLE TODAY — the core invariant Plan B depends on, on the existing
   slot-indexed device path with NO new API: overwriting an adapter's A/B *in
   place* (same device address) is reflected on CUDA-graph replay WITHOUT
   re-capture.

2. API SPEC (skipped until implemented) — REAL test bodies (not pseudo-code)
   driving the proposed Plan B worker-extension API exactly as an RL framework
   would, via `llm._collective_rpc(...)`. These mirror
   tests/unittest/_torch/ray_orchestrator/single_gpu/test_llm_update_weights.py.

   Proposed Plan B API (to be added to tensorrt_llm.llmapi.rlhf_utils.WorkerExtension,
   backed by PeftCacheManager.update_task_peft / pin_task and LoraCache::updateWeights):
     - pin_lora_task(task_id)                       -> keep resident (R1)
     - get_lora_task_pointers(task_id)              -> list[(in_ptr, out_ptr)] per (layer,module)
     - update_lora_weights(ipc_handles, task_id)    -> in-place device->device overwrite (R2/R4/R7)
       followed by update_lora_weights(None, task_id) to finalize (mirrors update_weights).

The MoE-LoRA harness is imported from the sibling device-path test (matches the
cross-test-import convention in tests/unittest/_torch).
"""

import base64
import pickle

import pytest
import torch

from test_moe_lora_device_path import (  # noqa: E402
    _ATOL, _RTOL, _build_base_inputs, _call_fused_moe, _make_adapter_set,
    _reference, _slot_kwargs, _warmup_and_capture, requires_cuda_and_op)

# Plan B engine API does not exist yet; these tests document the intended
# contract. Remove this marker (per-test) as each piece of the API lands.
plan_b_not_implemented = pytest.mark.skip(
    reason="Plan B in-place refit API (WorkerExtension.update_lora_weights / "
    "pin_lora_task / get_lora_task_pointers, PeftCacheManager.update_task_peft, "
    "LoraCache::updateWeights) is not implemented yet — TDD/API spec.")

# Routed-expert MoE LoRA target modules (TRT-LLM module names).
_MOE_LORA_MODULES = ["moe_h_to_4h", "moe_4h_to_h", "moe_gate"]


@pytest.fixture(autouse=True)
def _isolate_moe_runner_cache():
    """Fresh cached FusedMoeRunner per test; release captured graphs after
    (mirrors the device-path test)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    yield
    MoERunner.runner_dict.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _overwrite_adapter_in_place(dst, src):
    """Copy src's A/B into dst's existing buffers (same data_ptr). Python
    stand-in for what LoraCache::updateWeights will do: overwrite the resident
    device pages, leaving the weight pointers unchanged."""
    for key in ("fc1", "gated", "fc2"):
        before_a, before_b = dst[key]["A"].data_ptr(), dst[key]["B"].data_ptr()
        dst[key]["A"].copy_(src[key]["A"])
        dst[key]["B"].copy_(src[key]["B"])
        assert dst[key]["A"].data_ptr() == before_a, "in-place copy must not realloc A"
        assert dst[key]["B"].data_ptr() == before_b, "in-place copy must not realloc B"


# --------------------------------------------------------------------------- #
# 1. RUNNABLE — the core Plan B invariant on existing code
# --------------------------------------------------------------------------- #
@requires_cuda_and_op
def test_inplace_overwrite_reflected_on_cuda_graph_replay():
    """Plan B foundation: overwriting an adapter's weights IN PLACE (same device
    address) is reflected on CUDA-graph replay WITHOUT re-capture.

    Capture the slot-indexed device path with weights v0, copy a *different*
    adapter set into the same buffers, then replay the SAME graph. Replay must
    now produce the reference output for the NEW weights — proving the kernel
    reads whatever bytes live behind the (unchanged) slot pointer.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device)
    adapters = _make_adapter_set(num_experts, rank, hidden_size, inter_size,
                                 dtype, device, base_seed=400)
    new_adapters = _make_adapter_set(num_experts, rank, hidden_size, inter_size,
                                     dtype, device, base_seed=999)

    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32)
    slot_kwargs = _slot_kwargs(token_to_slot, [adapters], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype,
                               dict(slot_kwargs))

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    out_old = captured.clone()
    torch.testing.assert_close(
        out_old, _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters),
        rtol=_RTOL, atol=_ATOL)

    # Refit: overwrite the resident buffers in place (pointers unchanged), then
    # replay the SAME captured graph — no re-capture, no new slot, no new id.
    _overwrite_adapter_in_place(adapters, new_adapters)
    graph.replay()
    torch.cuda.synchronize()
    out_new = captured.clone()

    assert torch.isfinite(out_new).all()
    torch.testing.assert_close(
        out_new, _reference(x, w3_w1, w2, topk_ids, topk_scores, new_adapters),
        rtol=_RTOL, atol=_ATOL)
    assert not torch.allclose(out_new, out_old, rtol=_RTOL, atol=_ATOL)


# --------------------------------------------------------------------------- #
# 2. API SPEC (skipped) — real bodies driving the proposed Plan B API
# --------------------------------------------------------------------------- #
def _adapter_lora_dir(tmp_path, *, rank, num_experts, hidden, inter, seed):
    """Write a per-expert MoE LoRA adapter (PEFT layout) to disk and return the
    dir. Real adapters for the moe_* modules are stacked [num_experts, ...];
    make_per_expert_lora produces exactly that shape (see moe_layout.py)."""
    from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora
    import safetensors.torch
    adir = tmp_path / f"moe_lora_r{rank}_s{seed}"
    adir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for module, (in_d, out_d) in {
            "moe_h_to_4h": (hidden, inter),
            "moe_gate": (hidden, inter),
            "moe_4h_to_h": (inter, hidden)}.items():
        a = make_per_expert_lora(num_experts, rank, in_d, out_d, seed=seed)
        tensors[f"{module}.lora_A.weight"] = a["A"]   # [E, rank, in]
        tensors[f"{module}.lora_B.weight"] = a["B"]   # [E, out, rank]
    safetensors.torch.save_file(tensors, str(adir / "adapter_model.safetensors"))
    (adir / "adapter_config.json").write_text(
        f'{{"r": {rank}, "lora_alpha": {rank}, '
        f'"target_modules": {_MOE_LORA_MODULES}}}')
    return str(adir)


def _adapter_ipc_handles(named_tensors_by_device):
    """Serialize per-device adapter weight shards as CUDA IPC handles, exactly
    like RefHFModelWithIPCHandles.get_weight_ipc_handles_serialized in
    test_llm_update_weights.py — but for adapter A/B instead of base params.

    named_tensors_by_device: {cuda_device_idx: [(weight_name, tensor_on_that_device), ...]}
      where each tensor is that inference rank's EP/TP shard (the R7 reshard
      from the trainer's layout happens before this call).
    """
    from torch.multiprocessing.reductions import reduce_tensor

    from tensorrt_llm._torch.utils import get_device_uuid
    out = {}
    for dev, named in named_tensors_by_device.items():
        handles = [(name, reduce_tensor(t)) for name, t in named]
        out[get_device_uuid(dev)] = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
    return out


@plan_b_not_implemented
def test_update_lora_weights_keeps_task_id_and_pointers_stable(tmp_path):
    """SPEC: an IPC in-place refit overwrites the resident adapter's weights with
    the SAME task_id and the SAME device pointers (in-buffer update, R2/R5)."""
    from tensorrt_llm import LLM
    from tensorrt_llm.executor.request import LoRARequest
    from tensorrt_llm.llmapi import LoraConfig, MoeConfig, SamplingParams
    from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora
    from utils.llm_data import llm_models_root

    model_dir = str(llm_models_root() / "Qwen3" / "Qwen3-30B-A3B")
    rank, task_id = 16, 1
    adir = _adapter_lora_dir(tmp_path, rank=rank, num_experts=128, hidden=2048,
                             inter=768, seed=0)

    with LLM(model=model_dir,
             ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
             lora_config=LoraConfig(lora_target_modules=_MOE_LORA_MODULES,
                                    max_lora_rank=rank, max_loras=8, max_cpu_loras=8),
             moe_config=MoeConfig(backend="CUTLASS"),
             model_kwargs={"num_hidden_layers": 1}) as llm:
        req = LoRARequest("adapter-0", task_id, lora_path=adir)
        # First generate forces the adapter resident in a device slot.
        llm.generate(["hello"], SamplingParams(max_tokens=1), lora_request=req)

        llm._collective_rpc("pin_lora_task", (task_id, ))          # R1
        ptrs_before = llm._collective_rpc("get_lora_task_pointers", (task_id, ))

        # New adapter values, sharded per inference device (single GPU here).
        new = {0: []}
        for module, (in_d, out_d) in {"moe_h_to_4h": (2048, 768),
                                      "moe_gate": (2048, 768),
                                      "moe_4h_to_h": (768, 2048)}.items():
            a = make_per_expert_lora(128, rank, in_d, out_d, seed=999,
                                     device=torch.device("cuda:0"))
            new[0].append((f"{module}.lora_A.weight", a["A"]))
            new[0].append((f"{module}.lora_B.weight", a["B"]))
        ipc = _adapter_ipc_handles(new)

        llm._collective_rpc("update_lora_weights", (ipc, task_id))   # R2: in-place
        llm._collective_rpc("update_lora_weights", (None, task_id))  # finalize

        ptrs_after = llm._collective_rpc("get_lora_task_pointers", (task_id, ))
        # In-buffer update: same task_id, identical device pointers.
        assert ptrs_after == ptrs_before


@plan_b_not_implemented
def test_update_lora_weights_rejects_rank_change(tmp_path):
    """SPEC: refit must preserve the loaded rank (R3); a rank change would force
    CUDA-graph recapture + repaging and is rejected."""
    from tensorrt_llm import LLM
    from tensorrt_llm.executor.request import LoRARequest
    from tensorrt_llm.llmapi import LoraConfig, MoeConfig, SamplingParams
    from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora
    from utils.llm_data import llm_models_root

    model_dir = str(llm_models_root() / "Qwen3" / "Qwen3-30B-A3B")
    task_id = 1
    adir = _adapter_lora_dir(tmp_path, rank=8, num_experts=128, hidden=2048,
                             inter=768, seed=0)
    with LLM(model=model_dir,
             ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
             lora_config=LoraConfig(lora_target_modules=_MOE_LORA_MODULES,
                                    max_lora_rank=16, max_loras=8, max_cpu_loras=8),
             moe_config=MoeConfig(backend="CUTLASS"),
             model_kwargs={"num_hidden_layers": 1}) as llm:
        req = LoRARequest("adapter-0", task_id, lora_path=adir)  # loaded at rank 8
        llm.generate(["hello"], SamplingParams(max_tokens=1), lora_request=req)
        llm._collective_rpc("pin_lora_task", (task_id, ))

        rank16 = {0: []}  # rank-16 weights for a rank-8 resident adapter
        for module, (in_d, out_d) in {"moe_h_to_4h": (2048, 768),
                                      "moe_gate": (2048, 768),
                                      "moe_4h_to_h": (768, 2048)}.items():
            a = make_per_expert_lora(128, 16, in_d, out_d, seed=1,
                                     device=torch.device("cuda:0"))
            rank16[0].append((f"{module}.lora_A.weight", a["A"]))
            rank16[0].append((f"{module}.lora_B.weight", a["B"]))
        with pytest.raises((ValueError, RuntimeError), match="rank"):
            llm._collective_rpc("update_lora_weights",
                                (_adapter_ipc_handles(rank16), task_id))


@plan_b_not_implemented
def test_inplace_refit_changes_generation_same_lora_id(tmp_path):
    """SPEC (end-to-end): after an in-place refit, generation for the SAME lora
    id reflects the new weights — no new LoRARequest id, no disk reload. Verifies
    with logits like test_llm_update_weights.compare_logits."""
    from tensorrt_llm import LLM
    from tensorrt_llm.executor.request import LoRARequest
    from tensorrt_llm.llmapi import LoraConfig, MoeConfig, SamplingParams
    from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora
    from utils.llm_data import llm_models_root

    model_dir = str(llm_models_root() / "Qwen3" / "Qwen3-30B-A3B")
    rank, task_id = 16, 1
    adir = _adapter_lora_dir(tmp_path, rank=rank, num_experts=128, hidden=2048,
                             inter=768, seed=0)
    sp = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=8)
    with LLM(model=model_dir,
             ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
             lora_config=LoraConfig(lora_target_modules=_MOE_LORA_MODULES,
                                    max_lora_rank=rank, max_loras=8, max_cpu_loras=8),
             moe_config=MoeConfig(backend="CUTLASS"),
             model_kwargs={"num_hidden_layers": 1}) as llm:
        req = LoRARequest("adapter-0", task_id, lora_path=adir)   # one fixed id
        out0 = llm.generate(["The capital of France is"], sp, lora_request=req)

        new = {0: []}
        for module, (in_d, out_d) in {"moe_h_to_4h": (2048, 768),
                                      "moe_gate": (2048, 768),
                                      "moe_4h_to_h": (768, 2048)}.items():
            a = make_per_expert_lora(128, rank, in_d, out_d, seed=12345,
                                     device=torch.device("cuda:0"))
            new[0].append((f"{module}.lora_A.weight", a["A"]))
            new[0].append((f"{module}.lora_B.weight", a["B"]))
        llm._collective_rpc("update_lora_weights", (_adapter_ipc_handles(new), task_id))
        llm._collective_rpc("update_lora_weights", (None, task_id))

        out1 = llm.generate(["The capital of France is"], sp, lora_request=req)
        # Same adapter id, but the refit changed the served weights -> output moves.
        assert req.lora_int_id == task_id
        l0 = out0[0].outputs[0].generation_logits
        l1 = out1[0].outputs[0].generation_logits
        assert not torch.allclose(l0, l1)
