# Generate text asynchronously with Ray orchestrator.
import asyncio

import ray

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._tmp_utils import (analyze_average_timestamps,
                                     dump_timestamps_to_json,
                                     print_enqueue_statistics,
                                     print_fetch_statistics)
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    # Configure KV cache memory usage fraction.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                    max_tokens=4096,
                                    enable_block_reuse=True)

    # model could accept HF model name or a path to local HF model.
    llm = LLM(
        model="/scratch/llm-models/llama-3.2-models/Llama-3.2-3B-Instruct-FP8",
        # model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        kv_cache_config=kv_cache_config,
        max_seq_len=1024,
        # max_batch_size=1,
        orchestrator_type="ray",  # Enable Ray orchestrator
        # Enable 2-way tensor parallelism
        # tensor_parallel_size=2
    )

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ] * 1000

    #* 100

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Collect all timestamps
    all_timestamps = []

    # Async based on Python coroutines
    async def task(prompt: str):
        output = await llm.generate_async(prompt, sampling_params)

        if output.outputs[0].timestamps:
            all_timestamps.append(output.outputs[0].timestamps)

        # print(
        #     f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        # )

    async def main():
        tasks = [task(prompt) for prompt in prompts]
        await asyncio.gather(*tasks)

    asyncio.run(main())

    analyze_average_timestamps(all_timestamps)
    dump_timestamps_to_json(all_timestamps, "timestamps_output.json")

    if hasattr(llm._executor, 'enqueue_timings'):
        print_enqueue_statistics(llm._executor.enqueue_timings)

    if hasattr(llm._executor, 'workers'):
        for i, worker in enumerate(llm._executor.workers):
            try:
                stats = worker.call_worker_method.remote('get_fetch_statistics')
                result = ray.get(stats)
                if result:
                    print_fetch_statistics(result['num_fetched_requests'],
                                           result['fetch_call_count'],
                                           rank=result['rank'])
            except Exception as e:
                print(f"Could not get fetch statistics from worker {i}: {e}")

    # Got output like follows:
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'


if __name__ == '__main__':
    main()
