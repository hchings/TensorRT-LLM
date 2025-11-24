import os
import sys

import pytest

# Add path for test utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


@pytest.mark.asyncio
async def test_async_llm_release_resume_memory(process_gpu_memory_info_available):
    """Test AsyncLLM's release_memory_async and resume_memory_async methods.

    This test verifies that:
    1. Memory is released when calling release_memory_async()
    2. Memory is restored when calling resume_memory_async()
    3. Generation works correctly before and after memory release/resume
    4. Generated outputs are consistent (deterministic with temperature=0)
    """
    llama_model_path = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=4096)

    llm = AsyncLLM(
        model=llama_model_path,
        enable_sleep=True,
        cuda_graph_config=None,  # CUDA Graph unsupported with sleep
        kv_cache_config=kv_cache_config,
    )

    # Setup the AsyncLLM
    await llm.setup_async()

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0)

    try:
        # Generate initial outputs to verify LLM works
        outputs = llm.generate(prompts, sampling_params)
        generated_before_sleep = [output.outputs[0].text for output in outputs]

        # Record memory usage in active state
        memory_usage_active = get_current_process_gpu_memory(True)

        # Release memory asynchronously
        await llm.release_memory_async()

        # Record memory usage after release
        memory_usage_released = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_released < memory_usage_active, (
                f"Memory should be released: {memory_usage_released} >= {memory_usage_active}"
            )

        # Resume memory asynchronously
        await llm.resume_memory_async()

        # Record memory usage after resume
        memory_usage_resumed = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_resumed > memory_usage_released, (
                f"Memory should be resumed: {memory_usage_resumed} <= {memory_usage_released}"
            )

        # Generate outputs again to verify LLM still works correctly
        outputs = llm.generate(prompts, sampling_params)
        generated_after_sleep = [output.outputs[0].text for output in outputs]

        # Verify outputs match (generation should be deterministic with temperature=0)
        for before, after in zip(generated_before_sleep, generated_after_sleep, strict=True):
            assert before == after, "Generated result mismatch before and after sleep"

    finally:
        # Clean up
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_release_resume_with_async_generation(process_gpu_memory_info_available):
    """Test AsyncLLM memory management with async generation.

    This test verifies that release_memory_async and resume_memory_async work
    correctly with async generation patterns.
    """
    llama_model_path = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=4096)

    llm = AsyncLLM(
        model=llama_model_path,
        enable_sleep=True,
        cuda_graph_config=None,
        kv_cache_config=kv_cache_config,
    )

    # Setup the AsyncLLM
    await llm.setup_async()

    prompt = "The capital of France is"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    try:
        # Generate using async interface
        output_before = await llm.generate_async(prompt, sampling_params)
        text_before = output_before.outputs[0].text

        # Record memory usage
        memory_usage_active = get_current_process_gpu_memory(True)

        # Release and resume memory
        await llm.release_memory_async()
        memory_usage_released = get_current_process_gpu_memory(True)

        await llm.resume_memory_async()
        memory_usage_resumed = get_current_process_gpu_memory(True)

        # Verify memory changes
        if process_gpu_memory_info_available:
            assert memory_usage_released < memory_usage_active, (
                f"Memory should be released: {memory_usage_released} >= {memory_usage_active}"
            )
            assert memory_usage_resumed > memory_usage_released, (
                f"Memory should be resumed: {memory_usage_resumed} <= {memory_usage_released}"
            )

        # Generate again using async interface
        output_after = await llm.generate_async(prompt, sampling_params)
        text_after = output_after.outputs[0].text

        # Verify generation consistency
        assert text_before == text_after, (
            f"Generated text mismatch: '{text_before}' != '{text_after}'"
        )

    finally:
        # Clean up
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_multiple_release_resume_cycles(process_gpu_memory_info_available):
    """Test multiple cycles of memory release and resume.

    This test verifies that AsyncLLM can handle multiple consecutive
    release/resume cycles without issues.
    """
    llama_model_path = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=4096)

    llm = AsyncLLM(
        model=llama_model_path,
        enable_sleep=True,
        cuda_graph_config=None,
        kv_cache_config=kv_cache_config,
    )

    # Setup the AsyncLLM
    await llm.setup_async()

    prompt = "A B C"
    sampling_params = SamplingParams(temperature=0, max_tokens=8)

    try:
        # Get baseline output
        baseline_output = llm.generate([prompt], sampling_params)
        baseline_text = baseline_output[0].outputs[0].text

        # Perform multiple release/resume cycles
        num_cycles = 3
        for cycle in range(num_cycles):
            # Release memory
            await llm.release_memory_async()

            if process_gpu_memory_info_available:
                memory_after_release = get_current_process_gpu_memory(True)

            # Resume memory
            await llm.resume_memory_async()

            if process_gpu_memory_info_available:
                memory_after_resume = get_current_process_gpu_memory(True)
                assert memory_after_resume > memory_after_release, (
                    f"Cycle {cycle + 1}: Memory not properly resumed"
                )

            # Verify generation still works
            output = llm.generate([prompt], sampling_params)
            text = output[0].outputs[0].text
            assert text == baseline_text, f"Cycle {cycle + 1}: Generated text mismatch"

    finally:
        # Clean up
        llm.shutdown()
