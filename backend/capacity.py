from __future__ import annotations

import math


def capacity_plan(
    videos: int = 50_000,
    hours: float = 1.0,
    average_megabytes: float = 12.0,
    download_gbps: float = 1.0,
    workers: int = 16,
    seconds_per_video_per_worker: float = 4.0,
) -> dict:
    window = max(hours * 3600, 1)
    required_vps = videos / window
    total_gb = videos * average_megabytes / 1000
    payload_gbps = total_gb * 8 / (hours * 3600)
    # Allow 12% for TCP/TLS/container overhead and retries.
    required_link_gbps = payload_gbps / .88
    download_seconds = total_gb * 8 / max(download_gbps * .88, .001)
    compute_vps = workers / max(seconds_per_video_per_worker, .001)
    required_workers = math.ceil(required_vps * seconds_per_video_per_worker)
    return {
        "target": {"videos": videos, "hours": hours, "videos_per_second": round(required_vps, 3)},
        "storage": {"average_video_mb": average_megabytes, "total_payload_gb": round(total_gb, 2)},
        "network": {"configured_gbps": download_gbps, "minimum_sustained_gbps": round(required_link_gbps, 3), "estimated_download_minutes": round(download_seconds / 60, 1), "feasible": download_seconds <= window},
        "compute": {"workers": workers, "measured_seconds_per_video": seconds_per_video_per_worker, "capacity_videos_per_second": round(compute_vps, 3), "minimum_workers": required_workers, "feasible": compute_vps >= required_vps},
        "overall_feasible": download_seconds <= window and compute_vps >= required_vps,
        "warning": "This estimate assumes download and analysis overlap, model weights are pre-warmed, and storage sustains concurrent writes.",
    }
