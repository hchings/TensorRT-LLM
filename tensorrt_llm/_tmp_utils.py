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

    # Calculate each metric using actual timestamp names
    # Submit to enqueue: from executor submit to worker enqueue (Ray RPC or MPI IPC)
    if 'executor_submit_request' in timestamps and 'worker_enqueue_request' in timestamps:
        latencies['submit_request_to_enqueue'] = (
            timestamps['worker_enqueue_request'] -
            timestamps['executor_submit_request']) * 1000

    # Queue wait time: from queued to fetched
    if 'request_queued' in timestamps and 'request_fetched' in timestamps:
        latencies['queue_wait_time'] = (timestamps['request_fetched'] -
                                        timestamps['request_queued']) * 1000

    # Scheduling wait time: from fetched to scheduled (multi-iteration wait)
    if 'request_fetched' in timestamps and 'batch_scheduled_time' in timestamps:
        latencies['scheduling_wait_time'] = (
            timestamps['batch_scheduled_time'] -
            timestamps['request_fetched']) * 1000

    # Pre-forward overhead: from scheduled to forward step start (same iteration)
    if 'batch_scheduled_time' in timestamps and 'forward_step_start' in timestamps:
        latencies['pre_forward_overhead'] = (
            timestamps['forward_step_start'] -
            timestamps['batch_scheduled_time']) * 1000

    # Forward step time: actual GPU compute
    if 'forward_step_start' in timestamps and 'forward_step_end' in timestamps:
        latencies['forward_step_time'] = (
            timestamps['forward_step_end'] -
            timestamps['forward_step_start']) * 1000

    # Post-processing time: from forward end to response created
    if 'forward_step_end' in timestamps and 'response_created' in timestamps:
        latencies['post_processing_time'] = (timestamps['response_created'] -
                                             timestamps['forward_step_end']) * 1000

    # Execution time: from fetched to response created (total execution)
    if 'request_fetched' in timestamps and 'response_created' in timestamps:
        latencies['execution_time'] = (timestamps['response_created'] -
                                       timestamps['request_fetched']) * 1000

    # Response handling: from response created to enqueued (worker-side processing)
    if 'response_created' in timestamps and 'response_enqueued' in timestamps:
        latencies['response_handling'] = (timestamps['response_enqueued'] -
                                          timestamps['response_created']) * 1000

    # Enqueue response to handle: from response enqueued to client receives (Ray RPC or MPI IPC)
    if 'response_enqueued' in timestamps and 'handle_response' in timestamps:
        latencies['enqueue_response_to_handle'] = (
            timestamps['handle_response'] -
            timestamps['response_enqueued']) * 1000

    # Total E2E latency: from executor submit to handle_response
    if 'executor_submit_request' in timestamps and 'handle_response' in timestamps:
        latencies['total_e2e'] = (timestamps['handle_response'] -
                                  timestamps['executor_submit_request']) * 1000

    # Calculate communication overhead (sum of both communication latencies: Ray RPC or MPI IPC)
    if all(k in timestamps for k in [
            'executor_submit_request', 'worker_enqueue_request',
            'response_enqueued', 'handle_response'
    ]):
        latencies['communication_overhead'] = (
            (timestamps['worker_enqueue_request'] -
             timestamps['executor_submit_request']) +
            (timestamps['handle_response'] -
             timestamps['response_enqueued'])) * 1000

    return latencies


def analyze_average_timestamps(all_timestamps):
    """
    Calculate and print average latencies across all requests.
    all_timestamps: list of timestamp dicts from each request
    """
    if not is_timestamp_debug_enabled():
        return

    if not all_timestamps:
        print("No timestamps available")
        return

    mode = "[Ray]" if mpi_disabled() else "[MPI]"
    print(f"\n=== {mode} Analyzing {len(all_timestamps)} requests ===")

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
    print(f"\n=== Average Latency Breakdown (milliseconds) ===")

    metrics = [
        ('submit_request_to_enqueue', 'Submit to enqueue'),
        ('queue_wait_time', 'Request Queue wait time'),
        ('execution_time', 'Total execution time'),
        ('scheduling_wait_time', '  ├─ Scheduling wait time'),
        ('pre_forward_overhead', '  ├─ Pre-forward overhead'),
        ('forward_step_time', '  ├─ Forward step'),
        ('post_processing_time', '  └─ Post-processing time'),
        ('response_handling', 'Response handling'),
        ('enqueue_response_to_handle', 'Enqueue to handle'),
        ('total_e2e', 'Total E2E latency'),
        ('communication_overhead', 'Total communication overhead'),
    ]

    for metric_key, metric_name in metrics:
        # Add separator before E2E metrics
        if metric_key == 'total_e2e':
            print("  " + "-" * 68)

        values = [lat[metric_key] for lat in all_latencies if metric_key in lat]
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            # Calculate variance
            variance = sum((x - avg)**2 for x in values) / len(values)
            print(
                f"  {metric_name:32s}: {avg:8.3f} ms (min: {min_val:8.3f}, max: {max_val:9.3f}, var: {variance:10.3f})"
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

    print("=" * 70)
