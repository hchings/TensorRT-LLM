"""
Temporary utilities for timestamp analysis and Ray vs MPI latency comparison.
"""
import json

from tensorrt_llm._utils import mpi_disabled


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

    # Enqueue overhead: from worker enqueue to request queued
    if 'worker_enqueue_request' in timestamps and 'request_queued' in timestamps:
        latencies['enqueue_overhead'] = (
            timestamps['request_queued'] -
            timestamps['worker_enqueue_request']) * 1000

    # Queue wait time: from queued to fetched
    if 'request_queued' in timestamps and 'request_fetched' in timestamps:
        latencies['queue_wait_time'] = (timestamps['request_fetched'] -
                                        timestamps['request_queued']) * 1000

    # Execution time: from fetched to response created (actual GPU execution)
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
        ('submit_request_to_enqueue', 'Submit to enqueue (comm)'),
        ('enqueue_overhead', 'Enqueue overhead'),
        ('queue_wait_time', 'Queue wait time'),
        ('execution_time', 'Execution (actual GPU)'),
        ('response_handling', 'Response handling'),
        ('enqueue_response_to_handle', 'Enqueue to handle (comm)'),
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
    if not all_timestamps:
        print("No timestamps to dump")
        return

    print(
        f"\nDumping {len(all_timestamps)} timestamp records to {output_file}..."
    )
    with open(output_file, 'w') as f:
        json.dump(all_timestamps, f, indent=2)
    print(f"Timestamps saved to {output_file}")
