"""
Temporary utilities for timestamp analysis and Ray vs MPI latency comparison.
"""
import json
import os
from collections import Counter

from tensorrt_llm._utils import mpi_disabled


def is_timestamp_debug_enabled():
    return os.environ.get('TIMESTAMP_DEBUG', '0') == '1'


def calculate_latencies(timestamps):
    """
    Calculate latency metrics from a single set of timestamps.
    Returns a dict of latencies in milliseconds, or None if timestamps missing.
    """
    if not timestamps:
        return None

    latencies = {}

    latencies['submit_request_to_enqueue'] = (
        timestamps['worker_enqueue_request'] -
        timestamps['executor_submit_request']) * 1000

    # only for the fetch
    latencies['queue_wait_time'] = (timestamps['request_fetched'] -
                                    timestamps['request_queued']) * 1000

    latencies['num_iterations'] = timestamps['num_iterations']
    latencies['scheduling_wait_time'] = timestamps['scheduling_wait_time']
    latencies['pre_forward_overhead'] = timestamps['pre_forward_overhead']
    latencies['forward_step_time'] = timestamps['forward_step_time']
    latencies['post_processing_time'] = timestamps['post_processing_time']

    latencies['execution_time'] = (timestamps['response_created'] -
                                   timestamps['request_fetched']) * 1000

    latencies['response_handling'] = (timestamps['response_enqueued'] -
                                      timestamps['response_created']) * 1000

    latencies['enqueue_response_to_handle'] = (
        timestamps['handle_response'] - timestamps['response_enqueued']) * 1000

    latencies['total_e2e'] = (timestamps['handle_response'] -
                              timestamps['executor_submit_request']) * 1000

    latencies['communication_overhead'] = (
        (timestamps['worker_enqueue_request'] -
         timestamps['executor_submit_request']) +
        (timestamps['handle_response'] -
         timestamps['response_enqueued'])) * 1000

    return latencies


def analyze_average_timestamps(all_timestamps):
    if not is_timestamp_debug_enabled():
        return

    if not all_timestamps:
        print("No timestamps available")
        return

    mode = "[Ray]" if mpi_disabled() else "[MPI]"
    # Calculate latencies for each request
    all_latencies = []
    for ts in all_timestamps:
        latencies = calculate_latencies(ts)
        if latencies:
            all_latencies.append(latencies)

    if not all_latencies:
        print("No valid latencies calculated")
        return

    # Calculate averages
    print(
        f"\n=== [{mode}] Latency Breakdown (milliseconds) - Average over {len(all_timestamps)} request ==="
    )

    # Print first 20 submit_request_to_enqueue values
    submit_to_enqueue_values = [
        lat['submit_request_to_enqueue'] for lat in all_latencies
        if 'submit_request_to_enqueue' in lat
    ]
    if submit_to_enqueue_values:
        first_20 = ', '.join(
            [f"{x:.2f}" for x in submit_to_enqueue_values[:20]])
        print(f"  Submit to enqueue (first 20, ms): {first_20}", flush=True)
        print(flush=True)

    metrics = [
        ('submit_request_to_enqueue', 'Submit to enqueue'),
        ('queue_wait_time', 'Request Queue wait (1st fetch)'),
        ('execution_time', 'Time in executor loop (sum of all iterations)'),
        ('scheduling_wait_time', '  ├─ Scheduling wait'),
        ('pre_forward_overhead', '  ├─ Pre-forward overhead'),
        ('forward_step_time', '  ├─ Forward step'),
        ('post_processing_time', '  └─ Post-processing'),
        ('response_handling', 'Response handling (once)'),
        ('enqueue_response_to_handle', 'Enqueue to handle (once)'),
        # ('num_iterations', 'Avg iterations per request'),
        # ('total_e2e', 'Total E2E latency'),
        # ('communication_overhead', 'Total communication overhead'),
    ]

    for metric_key, metric_name in metrics:
        if metric_key == 'num_iterations':
            print("")
        if metric_key == 'total_e2e':
            print("  " + "-" * 68)

        values = [lat[metric_key] for lat in all_latencies if metric_key in lat]
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            variance = sum((x - avg)**2 for x in values) / len(values)

            if metric_key == 'num_iterations':
                print(
                    f"  {metric_name:48s}: {avg:8.1f} (min: {min_val:8.1f}, max: {max_val:9.1f})"
                )
            else:
                print(
                    f"  {metric_name:48s}: {avg:8.3f} ms (min: {min_val:8.3f}, max: {max_val:9.3f}, var: {variance:10.3f})"
                )

    print("=" * 70)


def dump_timestamps_to_json(all_timestamps,
                            output_file="timestamps_output.json"):
    if not is_timestamp_debug_enabled():
        return

    if not all_timestamps:
        print("No timestamps to dump")
        return

    print(
        f"\nDumping {len(all_timestamps)} timestamp records to {output_file}..."
    )
    with open(output_file, 'w') as f:
        json.dump(all_timestamps, f, indent=2)
    print(f"Timestamps saved to {output_file}")


def print_fetch_statistics(num_fetched_requests, fetch_call_count, rank=None):
    if not is_timestamp_debug_enabled():
        return

    if not num_fetched_requests:
        return

    rank_str = f"[Rank {rank}]" if rank is not None else ""
    mode = "[Ray]" if mpi_disabled() else "[MPI]"

    print(f"\n=== {mode}{rank_str} Fetch Request Statistics ===")
    print(f"  Total fetch calls: {fetch_call_count}")

    size_distribution = Counter(num_fetched_requests)
    print(f"\n  Fetch Size Distribution:")
    for size in sorted(size_distribution.keys()):
        count = size_distribution[size]
        percentage = (count / len(num_fetched_requests)) * 100
        print(f"    {size:3d} requests: {count:5d} times ({percentage:5.1f}%)")

    print(f"\n  Num fetched requests (all iterations): {num_fetched_requests}")

    print("=" * 70)


def print_enqueue_statistics(enqueue_timings):
    if not is_timestamp_debug_enabled():
        return

    if not enqueue_timings:
        return

    mode = "[Ray]" if mpi_disabled() else "[MPI]"
    num_requests = len(enqueue_timings)

    print(
        f"\n=== {mode} Enqueue Request Timing Statistics ({num_requests} requests) ==="
    )
    first_20_enqueue = ', '.join([f"{x:.2f}" for x in enqueue_timings[:20]])
    print(f"  Direct enqueue (first 20, ms): {first_20_enqueue}", flush=True)

    avg = sum(enqueue_timings) / num_requests
    min_val = min(enqueue_timings)
    max_val = max(enqueue_timings)

    # Calculate percentiles
    sorted_timings = sorted(enqueue_timings)
    p10 = sorted_timings[int(num_requests *
                             0.1)] if num_requests > 1 else sorted_timings[0]
    p50 = sorted_timings[num_requests // 2]
    p90 = sorted_timings[int(num_requests * 0.9)]

    print(f"  Avg: {avg:.2f} ms")
    print(f"  Min: {min_val:.2f} ms")
    print(f"  Max: {max_val:.2f} ms")
    print(f"  P10: {p10:.2f} ms")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P90: {p90:.2f} ms")
    print("=" * 70)
