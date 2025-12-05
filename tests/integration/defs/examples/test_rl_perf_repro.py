import os
import re

import pytest
from defs.common import venv_check_output
from defs.conftest import llm_models_root


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize(
    "tp_size, num_instances", [(1, 8), (4, 2)], ids=["tp1_instances8", "tp4_instances2"]
)
def test_rl_perf_repro(llm_root, ray_example_root, llm_venv, tp_size, num_instances):
    if tp_size == 4:
        max_batch_size = 1024
    else:
        max_batch_size = 384
    script_path = os.path.join(ray_example_root, "rl_perf_repro.py")
    cmd = [
        script_path,
        "--tp_size",
        str(tp_size),
        "--num_instances",
        str(num_instances),
        "--top_p",
        "1",
        "--logprobs",
        "1",
        "--max_batch_size",
        str(max_batch_size),
    ]

    model_dir = f"{llm_models_root()}/Qwen2-7B-Instruct"
    cmd.extend(["--model_dir", model_dir])

    data_path = os.path.join(llm_root, "tests", "integration", "test_input_files", "prompts.json")
    cmd.extend(["--data_path", data_path])

    output = venv_check_output(llm_venv, cmd)

    # Extract Time taken from output
    match = re.search(r"Time taken:\s+([\d.]+)\s+seconds", output)
    assert match, f"Could not find 'Time taken' in output:\n{output}"
    time_taken = float(match.group(1))

    # Set expected baseline based on configuration
    if tp_size == 1 and num_instances == 8:
        expected_time = 14.75
    elif tp_size == 4 and num_instances == 2:
        expected_time = 23.35
    else:
        pytest.fail(f"Unknown configuration: tp_size={tp_size}, num_instances={num_instances}")

    # Verify performance is not regressed (slower) by more than 5%
    # Allow any improvement (faster), only fail if slower by >5%
    relative_diff = (time_taken - expected_time) / expected_time
    assert relative_diff <= 0.05, (
        f"Performance regression detected for tp{tp_size}_instances{num_instances}: "
        f"Time taken: {time_taken:.2f}s, Expected: {expected_time:.2f}s, "
        f"Slowdown: {relative_diff * 100:.2f}% (threshold: 5%)"
    )
