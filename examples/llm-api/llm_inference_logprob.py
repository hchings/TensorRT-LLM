import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._tensorrt_engine import LLM as TrtLLM


def main():
    llm = LLM(
        model="/scratch/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        orchestrator_type="ray"
    )

    # llm = TrtLLM(
    #     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # )

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # logprobs behavior (follows vLLM/OpenAI API):
    # - logprobs=None: No log probabilities returned
    # - logprobs=0: Returns only the sampled token's logprob (1 element)
    # - logprobs=K (K>0): Returns sampled token (first) + top-K tokens (up to K+1 elements)
    # - logprobs=-1: Returns sampled token + all vocab_size tokens
    sampling_params = SamplingParams(
        max_tokens=10,
        # temperature=0.7,
        # top_p=0.95,
        logprobs=0,  # Set to 0 for sampled only, or K>0 for sampled + top-K
    )

    for output in llm.generate(prompts, sampling_params):
        print(f"\n{'='*80}")
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Generated token IDs: {output.outputs[0].token_ids}")

        if output.outputs[0].generation_logits is not None:
            logits = output.outputs[0].generation_logits

            # sanity check on sampled logits
            num_logits = logits.shape[0]
            sampled_logits = [logits[i, token_id].item() for i, token_id in enumerate(output.outputs[0].token_ids[:num_logits])]
            print(f"Logits of sampled tokens: {sampled_logits}")

        if output.outputs[0].logprobs:
            print(f"\nLogprobs for each generated token:")
            for i, (token_id, token_logprobs) in enumerate(
                zip(output.outputs[0].token_ids, output.outputs[0].logprobs)
            ):
                print(f"\n  Token {i}: ID={token_id}, Text={llm.tokenizer.decode([token_id])!r}")

                # Verify vLLM/OpenAI API behavior:
                # - logprobs=0: should have only 1 entry (sampled token)
                # - logprobs=K (K>0): should have up to K+1 entries (sampled + top-K, with dedup)
                if sampling_params.logprobs == 0:
                    assert len(token_logprobs) == 1, f"Expected 1 logprob for sampled token, got {len(token_logprobs)}"
                elif sampling_params.logprobs is not None and sampling_params.logprobs > 0:
                    assert len(token_logprobs) <= sampling_params.logprobs + 1, \
                        f"Expected at most {sampling_params.logprobs + 1} logprobs, got {len(token_logprobs)}"
                assert token_id in token_logprobs, f"Sampled token {token_id} not in logprobs dict."

                for tid, logprob_obj in token_logprobs.items():
                    token_text = llm.tokenizer.decode([tid])
                    is_sampled = "← SAMPLED" if tid == token_id else ""
                    print(f"    • Token {tid:5d} ({token_text:15s}): "
                          f"logprob={logprob_obj.logprob:8.4f}, "
                          f"rank={logprob_obj.rank} {is_sampled}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
