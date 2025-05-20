"""
Utility functions and helpers for the project.
- Includes device/scaler setup and CPU detection for SLURM or local environments.
- Designed for use in a public repository (no hardcoded paths).
"""
import os
import sys
import subprocess
import torch


# Device and AMP scaler setup
# Use CUDA if available, else fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler()


def detect_aval_cpus():
    """
    Detect the number of available CPUs for data loading or parallel processing.
    - If running under SLURM, attempts to detect allocated CPUs for the current job.
    - Otherwise, falls back to the number of CPUs available to the process.
    Returns:
        int: Number of available CPUs (at least 1)
    """
    try:
        # Try to get the current SLURM job ID
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        # Query SLURM for CPUs allocated to this job
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        # Also check the number of CPUs available to the process
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except Exception:
        # Fallback: use a single CPU if detection fails
        cpus = 1  # or os.cpu_count()
    return cpus

