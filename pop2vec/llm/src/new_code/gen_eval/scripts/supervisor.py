#!/usr/bin/env python3
"""
Pipeline Supervisor for Generative Evaluation

A long-running daemon that manages the entire evaluation pipeline:
- Submits generation jobs (GPU)
- Monitors job status via squeue and log files
- Automatically submits statistics jobs when generation completes
- Detects failures and optionally resubmits
- Writes human-readable dashboard and machine-readable state

Designed to run as a SLURM job on the work_env partition (login node).

Usage:
    # Submit as SLURM job (recommended)
    sbatch supervisor.slurm
    
    # Or run directly (will stop if terminal closes)
    python supervisor.py --config supervisor_config.yaml

Configuration (supervisor_config.yaml):
    models:
      - model_v1_gen_20251117
      - model_v2
    experiments:
      - exp_n10_c100_h20_g100
      - exp_n100_c100_h20_g100
    gpus:
      ossc9424vm1: [0, 1, 2, 3]
      ossc9424vm2: [0, 1, 2, 3]  # Optional second node
    max_retries: 2
    poll_interval_seconds: 60
    auto_submit_stats: true
    auto_resubmit_failed: false  # Set true to auto-retry failures
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


# ============================================================================
# Configuration
# ============================================================================

# Default paths - adjust these for your environment
DEFAULT_SLURM_DIR = Path(__file__).parent.parent / "slurm_scripts"
DEFAULT_OUTPUT_DIR = Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval")
DEFAULT_LOG_DIR = Path("/projects/0/prjs1589/stonybrook/logs")
DEFAULT_STATE_DIR = Path(__file__).parent.parent / "supervisor_state"


class JobStatus(Enum):
    """Status of a pipeline job."""
    NOT_STARTED = "not_started"
    QUEUED = "queued"        # Submitted, waiting in queue
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Finished with error
    CANCELLED = "cancelled"  # Manually cancelled


@dataclass
class JobState:
    """State of a single job."""
    model: str
    experiment: str
    job_type: str  # 'gen', 'stats', 'plot'
    status: JobStatus = JobStatus.NOT_STARTED
    slurm_job_id: Optional[str] = None
    gpu_slot: Optional[str] = None  # "node:gpu_idx" for gen jobs
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    depends_on: Optional[str] = None  # Job ID this depends on
    
    @property
    def key(self) -> str:
        return f"{self.job_type}_{self.model}_{self.experiment}"
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'JobState':
        d['status'] = JobStatus(d['status'])
        return cls(**d)


@dataclass
class PipelineState:
    """Complete state of the pipeline."""
    models: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    jobs: Dict[str, JobState] = field(default_factory=dict)
    gpu_assignments: Dict[str, str] = field(default_factory=dict)  # gpu_slot -> job_key
    last_update: Optional[str] = None
    start_time: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'models': self.models,
            'experiments': self.experiments,
            'jobs': {k: v.to_dict() for k, v in self.jobs.items()},
            'gpu_assignments': self.gpu_assignments,
            'last_update': self.last_update,
            'start_time': self.start_time,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PipelineState':
        state = cls(
            models=d.get('models', []),
            experiments=d.get('experiments', []),
            gpu_assignments=d.get('gpu_assignments', {}),
            last_update=d.get('last_update'),
            start_time=d.get('start_time'),
        )
        for k, v in d.get('jobs', {}).items():
            state.jobs[k] = JobState.from_dict(v)
        return state


# ============================================================================
# SLURM Interface
# ============================================================================

def run_command(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_slurm_jobs(user: str = None) -> Dict[str, Dict]:
    """
    Get all SLURM jobs for the user.
    
    Returns:
        Dict[job_id] = {'name': str, 'state': str, 'time': str, 'nodelist': str}
    """
    if user is None:
        user = os.environ.get('USER', 'unknown')
    
    # Format: job_id|name|state|time|nodelist
    cmd = ['squeue', '-u', user, '-h', '-o', '%i|%j|%T|%M|%N']
    rc, stdout, stderr = run_command(cmd)
    
    jobs = {}
    if rc == 0:
        for line in stdout.strip().split('\n'):
            if line:
                parts = line.split('|')
                if len(parts) >= 5:
                    job_id, name, state, runtime, nodelist = parts[:5]
                    jobs[job_id] = {
                        'name': name,
                        'state': state,  # PENDING, RUNNING, COMPLETING, etc.
                        'time': runtime,
                        'nodelist': nodelist,
                    }
    return jobs


def submit_job(script_path: str, dependency: str = None) -> Optional[str]:
    """
    Submit a SLURM job and return the job ID.
    
    Args:
        script_path: Path to SLURM script
        dependency: Optional dependency string (e.g., "afterok:12345")
    
    Returns:
        Job ID if successful, None otherwise
    """
    cmd = ['sbatch']
    if dependency:
        cmd.append(f'--dependency={dependency}')
    cmd.append(script_path)
    
    rc, stdout, stderr = run_command(cmd)
    
    if rc == 0:
        # Parse "Submitted batch job 12345"
        match = re.search(r'Submitted batch job (\d+)', stdout)
        if match:
            return match.group(1)
    
    print(f"[ERROR] Failed to submit {script_path}: {stderr}")
    return None


def cancel_job(job_id: str) -> bool:
    """Cancel a SLURM job."""
    rc, _, _ = run_command(['scancel', job_id])
    return rc == 0


# ============================================================================
# File/Output Checking
# ============================================================================

def check_generation_complete(output_dir: Path, experiment: str) -> bool:
    """Check if generation output files exist."""
    exp_dir = output_dir / experiment
    required_files = [
        'generated_sequences.parquet',
        'original_sequences.parquet',
        'generation_metadata.json',
    ]
    return all((exp_dir / f).exists() for f in required_files)


def check_statistics_complete(output_dir: Path, experiment: str) -> bool:
    """Check if statistics output files exist."""
    exp_dir = output_dir / experiment
    required_files = [
        'token_counts.csv',
        'statistics_metadata.json',
    ]
    return all((exp_dir / f).exists() for f in required_files)


def parse_log_for_completion(log_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Parse a SLURM log file to check if job completed successfully.
    
    Returns:
        (is_complete, error_message)
    """
    if not log_path.exists():
        return False, None
    
    try:
        content = log_path.read_text()
        
        # Check for explicit completion marker
        if "Completed:" in content and "Generation Complete!" in content:
            return True, None
        if "Completed:" in content and "Statistics Complete!" in content:
            return True, None
        
        # Check for common errors
        error_patterns = [
            r'(CUDA out of memory)',
            r'(RuntimeError:.*)',
            r'(FileNotFoundError:.*)',
            r'(Exception:.*)',
            r'(Error:.*)',
            r'(Traceback \(most recent call last\))',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return False, match.group(1)[:200]  # Truncate long errors
        
        return False, None
    except Exception as e:
        return False, f"Could not read log: {e}"


def find_log_file(log_dir: Path, job_name: str, job_id: str = None) -> Optional[Path]:
    """Find the log file for a job."""
    # Pattern: jobname-jobid.out
    if job_id:
        log_path = log_dir / f"{job_name}-{job_id}.out"
        if log_path.exists():
            return log_path
    
    # Try to find any matching log
    pattern = f"{job_name}-*.out"
    matches = list(log_dir.glob(pattern))
    if matches:
        # Return most recent
        return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None


# ============================================================================
# Supervisor Core
# ============================================================================

class Supervisor:
    """Main supervisor class that manages the pipeline."""
    
    def __init__(
        self,
        models: List[str],
        experiments: List[str],
        gpus: Dict[str, List[int]],
        slurm_dir: Path,
        output_dir: Path,
        log_dir: Path,
        state_dir: Path,
        max_retries: int = 2,
        poll_interval: int = 60,
        auto_submit_stats: bool = True,
        auto_resubmit_failed: bool = False,
    ):
        self.models = models
        self.experiments = experiments
        self.gpus = gpus  # {'node': [0,1,2,3], ...}
        self.slurm_dir = Path(slurm_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.state_dir = Path(state_dir)
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.auto_submit_stats = auto_submit_stats
        self.auto_resubmit_failed = auto_resubmit_failed
        
        # Create state directory
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Build GPU slot list
        self.gpu_slots = []
        for node, indices in sorted(self.gpus.items()):
            for idx in sorted(indices):
                self.gpu_slots.append(f"{node}:{idx}")
        
        # Load or create state
        self.state_file = self.state_dir / "pipeline_state.json"
        self.dashboard_file = self.state_dir / "dashboard.txt"
        self.state = self._load_or_create_state()
        
        print(f"[SUPERVISOR] Initialized")
        print(f"  Models: {len(self.models)}")
        print(f"  Experiments: {len(self.experiments)}")
        print(f"  GPU Slots: {len(self.gpu_slots)} ({self.gpu_slots})")
        print(f"  Total Gen Jobs: {len(self.models) * len(self.experiments)}")
    
    def _load_or_create_state(self) -> PipelineState:
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                state = PipelineState.from_dict(data)
                print(f"[SUPERVISOR] Loaded existing state from {self.state_file}")
                return state
            except Exception as e:
                print(f"[SUPERVISOR] Could not load state: {e}, creating new")
        
        # Create new state
        state = PipelineState(
            models=self.models,
            experiments=self.experiments,
            start_time=datetime.now().isoformat(),
        )
        
        # Initialize all jobs
        for model in self.models:
            for exp in self.experiments:
                # Generation job
                gen_job = JobState(model=model, experiment=exp, job_type='gen')
                state.jobs[gen_job.key] = gen_job
                
                # Statistics job
                stats_job = JobState(model=model, experiment=exp, job_type='stats')
                state.jobs[stats_job.key] = stats_job
        
        return state
    
    def _save_state(self):
        """Save current state to file."""
        self.state.last_update = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def _get_available_gpu_slots(self) -> List[str]:
        """Get GPU slots that are not currently in use."""
        used_slots = set()
        for job in self.state.jobs.values():
            if job.status in [JobStatus.QUEUED, JobStatus.RUNNING] and job.gpu_slot:
                used_slots.add(job.gpu_slot)
        
        return [s for s in self.gpu_slots if s not in used_slots]
    
    def _get_script_path(self, model: str, exp: str, job_type: str) -> Optional[Path]:
        """Get the SLURM script path for a job."""
        prefix = 'gen' if job_type == 'gen' else 'stats'
        script_name = f"{prefix}_{model}_{exp}.sh"
        script_path = self.slurm_dir / script_name
        
        if script_path.exists():
            return script_path
        
        # Try without model name variations
        for f in self.slurm_dir.glob(f"{prefix}_*_{exp}.sh"):
            if model in f.name:
                return f
        
        return None
    
    def _update_job_from_slurm(self, slurm_jobs: Dict[str, Dict]):
        """Update job states based on current SLURM queue."""
        # Map SLURM job names to our jobs
        slurm_by_name = {info['name']: (jid, info) for jid, info in slurm_jobs.items()}
        
        for job in self.state.jobs.values():
            script_name = f"{job.job_type}_{job.model}_{job.experiment}"
            
            if job.slurm_job_id:
                # Check if our tracked job is still in queue
                if job.slurm_job_id in slurm_jobs:
                    info = slurm_jobs[job.slurm_job_id]
                    if info['state'] == 'RUNNING':
                        if job.status != JobStatus.RUNNING:
                            job.status = JobStatus.RUNNING
                            job.start_time = datetime.now().isoformat()
                    elif info['state'] == 'PENDING':
                        job.status = JobStatus.QUEUED
                else:
                    # Job no longer in queue - either completed or failed
                    if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                        # Need to check output files / logs
                        self._check_job_completion(job)
    
    def _check_job_completion(self, job: JobState):
        """Check if a job completed successfully or failed."""
        model_dir = self.output_dir / job.model
        
        if job.job_type == 'gen':
            if check_generation_complete(model_dir, job.experiment):
                job.status = JobStatus.COMPLETED
                job.end_time = datetime.now().isoformat()
                if job.start_time:
                    start = datetime.fromisoformat(job.start_time)
                    end = datetime.fromisoformat(job.end_time)
                    job.duration_seconds = (end - start).total_seconds()
                # Free up GPU slot
                if job.gpu_slot and job.gpu_slot in self.state.gpu_assignments:
                    if self.state.gpu_assignments[job.gpu_slot] == job.key:
                        del self.state.gpu_assignments[job.gpu_slot]
                return
        
        elif job.job_type == 'stats':
            if check_statistics_complete(model_dir, job.experiment):
                job.status = JobStatus.COMPLETED
                job.end_time = datetime.now().isoformat()
                return
        
        # Check log for errors
        job_name = f"{job.job_type}_{job.model}_{job.experiment}"
        log_file = find_log_file(self.log_dir, job_name, job.slurm_job_id)
        if log_file:
            is_complete, error = parse_log_for_completion(log_file)
            if error:
                job.status = JobStatus.FAILED
                job.error_message = error
                job.end_time = datetime.now().isoformat()
                # Free up GPU slot
                if job.gpu_slot:
                    self.state.gpu_assignments.pop(job.gpu_slot, None)
                return
        
        # If we can't determine, mark as failed
        job.status = JobStatus.FAILED
        job.error_message = "Job disappeared from queue without completion markers"
        job.end_time = datetime.now().isoformat()
        if job.gpu_slot:
            self.state.gpu_assignments.pop(job.gpu_slot, None)
    
    def _submit_generation_jobs(self):
        """Submit generation jobs for available GPU slots."""
        available_slots = self._get_available_gpu_slots()
        
        if not available_slots:
            return
        
        # Find jobs that need to be submitted
        pending_jobs = []
        for job in self.state.jobs.values():
            if job.job_type != 'gen':
                continue
            if job.status == JobStatus.NOT_STARTED:
                pending_jobs.append(job)
            elif job.status == JobStatus.FAILED and self.auto_resubmit_failed:
                if job.retry_count < self.max_retries:
                    pending_jobs.append(job)
        
        # Submit jobs for available slots
        for slot in available_slots:
            if not pending_jobs:
                break
            
            job = pending_jobs.pop(0)
            script_path = self._get_script_path(job.model, job.experiment, 'gen')
            
            if not script_path:
                print(f"[WARNING] No script found for {job.key}")
                continue
            
            # Check for GPU dependency (another job on same slot)
            dependency = None
            if slot in self.state.gpu_assignments:
                prev_job_key = self.state.gpu_assignments[slot]
                prev_job = self.state.jobs.get(prev_job_key)
                if prev_job and prev_job.slurm_job_id:
                    dependency = f"afterany:{prev_job.slurm_job_id}"
            
            # Submit
            job_id = submit_job(str(script_path), dependency)
            
            if job_id:
                job.slurm_job_id = job_id
                job.status = JobStatus.QUEUED
                job.gpu_slot = slot
                job.submit_time = datetime.now().isoformat()
                job.retry_count += 1
                self.state.gpu_assignments[slot] = job.key
                
                print(f"[SUBMIT] {job.key} -> Job {job_id} on {slot}")
                if dependency:
                    print(f"         Depends on: {dependency}")
    
    def _submit_statistics_jobs(self):
        """Submit statistics jobs for completed generation jobs."""
        if not self.auto_submit_stats:
            return
        
        for job in self.state.jobs.values():
            if job.job_type != 'stats':
                continue
            if job.status != JobStatus.NOT_STARTED:
                continue
            
            # Check if corresponding gen job is complete
            gen_key = f"gen_{job.model}_{job.experiment}"
            gen_job = self.state.jobs.get(gen_key)
            
            if not gen_job or gen_job.status != JobStatus.COMPLETED:
                continue
            
            script_path = self._get_script_path(job.model, job.experiment, 'stats')
            if not script_path:
                print(f"[WARNING] No stats script found for {job.key}")
                continue
            
            # Submit (no GPU needed for stats)
            job_id = submit_job(str(script_path))
            
            if job_id:
                job.slurm_job_id = job_id
                job.status = JobStatus.QUEUED
                job.submit_time = datetime.now().isoformat()
                print(f"[SUBMIT] {job.key} -> Job {job_id}")
    
    def _write_dashboard(self):
        """Write human-readable dashboard file."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"PIPELINE SUPERVISOR DASHBOARD")
        lines.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        gen_jobs = [j for j in self.state.jobs.values() if j.job_type == 'gen']
        stats_jobs = [j for j in self.state.jobs.values() if j.job_type == 'stats']
        
        def count_status(jobs, status):
            return len([j for j in jobs if j.status == status])
        
        lines.append("GENERATION JOBS:")
        lines.append(f"  Total: {len(gen_jobs)}")
        lines.append(f"  ✓ Completed: {count_status(gen_jobs, JobStatus.COMPLETED)}")
        lines.append(f"  ⟳ Running:   {count_status(gen_jobs, JobStatus.RUNNING)}")
        lines.append(f"  ⏳ Queued:    {count_status(gen_jobs, JobStatus.QUEUED)}")
        lines.append(f"  ✗ Failed:    {count_status(gen_jobs, JobStatus.FAILED)}")
        lines.append(f"  - Pending:   {count_status(gen_jobs, JobStatus.NOT_STARTED)}")
        lines.append("")
        
        lines.append("STATISTICS JOBS:")
        lines.append(f"  Total: {len(stats_jobs)}")
        lines.append(f"  ✓ Completed: {count_status(stats_jobs, JobStatus.COMPLETED)}")
        lines.append(f"  ⟳ Running:   {count_status(stats_jobs, JobStatus.RUNNING)}")
        lines.append(f"  ⏳ Queued:    {count_status(stats_jobs, JobStatus.QUEUED)}")
        lines.append(f"  ✗ Failed:    {count_status(stats_jobs, JobStatus.FAILED)}")
        lines.append(f"  - Pending:   {count_status(stats_jobs, JobStatus.NOT_STARTED)}")
        lines.append("")
        
        # GPU Slots
        lines.append("GPU SLOTS:")
        for slot in self.gpu_slots:
            job_key = self.state.gpu_assignments.get(slot, "-")
            if job_key != "-":
                job = self.state.jobs.get(job_key)
                status = job.status.value if job else "?"
                lines.append(f"  {slot}: {job_key} ({status})")
            else:
                lines.append(f"  {slot}: available")
        lines.append("")
        
        # Job Matrix (Model x Experiment)
        lines.append("JOB MATRIX (Generation):")
        lines.append("-" * 80)
        
        # Header
        header = f"{'Model':<30}"
        for exp in self.experiments:
            exp_short = exp.replace('exp_', '')[:15]
            header += f" {exp_short:^12}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Status symbols
        symbols = {
            JobStatus.NOT_STARTED: '-',
            JobStatus.QUEUED: '⏳',
            JobStatus.RUNNING: '⟳',
            JobStatus.COMPLETED: '✓',
            JobStatus.FAILED: '✗',
            JobStatus.CANCELLED: 'C',
        }
        
        for model in self.models:
            row = f"{model:<30}"
            for exp in self.experiments:
                key = f"gen_{model}_{exp}"
                job = self.state.jobs.get(key)
                if job:
                    sym = symbols.get(job.status, '?')
                    if job.duration_seconds:
                        dur = str(timedelta(seconds=int(job.duration_seconds)))
                        cell = f"{sym} {dur[:8]}"
                    else:
                        cell = sym
                else:
                    cell = '?'
                row += f" {cell:^12}"
            lines.append(row)
        
        lines.append("")
        lines.append("Legend: ✓=complete ⟳=running ⏳=queued ✗=failed -=pending")
        lines.append("")
        
        # Failed jobs details
        failed = [j for j in self.state.jobs.values() if j.status == JobStatus.FAILED]
        if failed:
            lines.append("FAILED JOBS:")
            lines.append("-" * 80)
            for job in failed:
                lines.append(f"  {job.key}")
                lines.append(f"    Error: {job.error_message or 'Unknown'}")
                lines.append(f"    Retries: {job.retry_count}/{self.max_retries}")
            lines.append("")
        
        # Timing statistics
        completed_gen = [j for j in gen_jobs if j.status == JobStatus.COMPLETED and j.duration_seconds]
        if completed_gen:
            lines.append("TIMING STATISTICS:")
            lines.append("-" * 80)
            durations = [j.duration_seconds for j in completed_gen]
            avg = sum(durations) / len(durations)
            lines.append(f"  Completed Gen Jobs: {len(completed_gen)}")
            lines.append(f"  Average Duration: {timedelta(seconds=int(avg))}")
            lines.append(f"  Min Duration: {timedelta(seconds=int(min(durations)))}")
            lines.append(f"  Max Duration: {timedelta(seconds=int(max(durations)))}")
            
            # Estimate remaining time
            remaining = len([j for j in gen_jobs if j.status in [JobStatus.NOT_STARTED, JobStatus.QUEUED]])
            if remaining > 0:
                # Account for parallelism
                parallel = len(self.gpu_slots)
                est_batches = (remaining + parallel - 1) // parallel
                est_time = avg * est_batches
                lines.append(f"  Remaining Jobs: {remaining}")
                lines.append(f"  Est. Time (with {parallel} GPUs): {timedelta(seconds=int(est_time))}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"State file: {self.state_file}")
        lines.append(f"To stop: scancel <supervisor_job_id>")
        lines.append("=" * 80)
        
        self.dashboard_file.write_text('\n'.join(lines))
    
    def run_once(self):
        """Run one iteration of the supervisor loop."""
        # Get current SLURM state
        slurm_jobs = get_slurm_jobs()
        
        # Update job states
        self._update_job_from_slurm(slurm_jobs)
        
        # Submit new generation jobs
        self._submit_generation_jobs()
        
        # Submit statistics jobs for completed generations
        self._submit_statistics_jobs()
        
        # Save state and dashboard
        self._save_state()
        self._write_dashboard()
    
    def run(self, max_iterations: int = None):
        """Run the supervisor loop."""
        print(f"[SUPERVISOR] Starting main loop (poll every {self.poll_interval}s)")
        print(f"[SUPERVISOR] Dashboard: {self.dashboard_file}")
        print(f"[SUPERVISOR] State: {self.state_file}")
        
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n[SUPERVISOR] === Iteration {iteration} at {datetime.now().strftime('%H:%M:%S')} ===")
                
                self.run_once()
                
                # Check if all done
                all_gen_done = all(
                    j.status in [JobStatus.COMPLETED, JobStatus.FAILED]
                    for j in self.state.jobs.values() if j.job_type == 'gen'
                )
                all_stats_done = all(
                    j.status in [JobStatus.COMPLETED, JobStatus.FAILED]
                    for j in self.state.jobs.values() if j.job_type == 'stats'
                )
                
                if all_gen_done and all_stats_done:
                    print("[SUPERVISOR] All jobs completed!")
                    self._write_dashboard()
                    break
                
                if max_iterations and iteration >= max_iterations:
                    print(f"[SUPERVISOR] Reached max iterations ({max_iterations})")
                    break
                
                # Sleep
                print(f"[SUPERVISOR] Sleeping {self.poll_interval}s...")
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print("\n[SUPERVISOR] Interrupted by user")
            self._save_state()
        
        print("[SUPERVISOR] Exiting")
        return self.state


def load_config(config_path: str) -> dict:
    """Load supervisor configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Supervisor for Generative Evaluation")
    parser.add_argument("--config", help="Path to supervisor config YAML")
    parser.add_argument("--models", nargs="+", help="Model names")
    parser.add_argument("--experiments", nargs="+", help="Experiment names")
    parser.add_argument("--gpus", help="GPU spec (e.g., 'ossc9424vm1:0,1,2,3')")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--max-iterations", type=int, help="Max iterations (for testing)")
    parser.add_argument("--slurm-dir", type=str, default=str(DEFAULT_SLURM_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--state-dir", type=str, default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--auto-resubmit", action="store_true", help="Auto-resubmit failed jobs")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries for failed jobs")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
        models = config.get('models', [])
        experiments = config.get('experiments', [])
        gpus = config.get('gpus', {'ossc9424vm1': [0, 1, 2, 3]})
        poll_interval = config.get('poll_interval_seconds', 60)
        auto_resubmit = config.get('auto_resubmit_failed', False)
        max_retries = config.get('max_retries', 2)
    else:
        models = args.models or []
        experiments = args.experiments or []
        gpus = {}
        if args.gpus:
            # Parse "node:0,1,2;node2:0,1"
            for part in args.gpus.split(';'):
                if ':' in part:
                    node, indices = part.split(':')
                    gpus[node] = [int(i) for i in indices.split(',')]
                else:
                    gpus['ossc9424vm1'] = [int(i) for i in part.split(',')]
        else:
            gpus = {'ossc9424vm1': [0, 1, 2, 3]}
        poll_interval = args.poll_interval
        auto_resubmit = args.auto_resubmit
        max_retries = args.max_retries
    
    if not models or not experiments:
        print("Error: Must specify models and experiments via --config or --models/--experiments")
        sys.exit(1)
    
    supervisor = Supervisor(
        models=models,
        experiments=experiments,
        gpus=gpus,
        slurm_dir=args.slurm_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        state_dir=args.state_dir,
        max_retries=max_retries,
        poll_interval=poll_interval,
        auto_submit_stats=True,
        auto_resubmit_failed=auto_resubmit,
    )
    
    supervisor.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
