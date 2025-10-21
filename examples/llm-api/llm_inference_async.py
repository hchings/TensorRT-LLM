### :section Basics
### :title Generate text asynchronously
### :order 1
import asyncio

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._tmp_utils import (analyze_average_timestamps,
                                     dump_timestamps_to_json,
                                     print_enqueue_statistics)
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    # model could accept HF model name or a path to local HF model.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                    max_tokens=4096,
                                    enable_block_reuse=True)

    llm = LLM(
        #model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model="/scratch/llm-models/llama-3.2-models/Llama-3.2-3B-Instruct-FP8",
        # tensor_parallel_size=2
        max_seq_len=1024,
        kv_cache_config=kv_cache_config
        # max_batch_size=1,
    )

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ] * 1000

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    all_timestamps = []

    # Async based on Python coroutines
    async def task(prompt: str):
        output = await llm.generate_async(prompt, sampling_params)

        if output.outputs[0].timestamps:
            all_timestamps.append(output.outputs[0].timestamps)

    async def main():
        tasks = [task(prompt) for prompt in prompts]
        await asyncio.gather(*tasks)

    asyncio.run(main())

    analyze_average_timestamps(all_timestamps)
    dump_timestamps_to_json(all_timestamps, "timestamps_output.json")

    print(
        f"executor type = {type(llm._executor)}, has enqueue_timings = {hasattr(llm._executor, 'enqueue_timings')}"
    )
    if hasattr(llm._executor, 'enqueue_timings'):
        print_enqueue_statistics(llm._executor.enqueue_timings)

    # Got output like follows:
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'


if __name__ == '__main__':
    main()
