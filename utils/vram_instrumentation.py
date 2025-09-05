
"""
VRAM Instrumentation Module
==========================
Modular VRAM monitoring utilities for Hugging Face, TRL, and Unsloth pipelines.
Provides comprehensive GPU memory tracking and reporting capabilities.
"""

import torch
import time
from typing import Dict, List, Optional, Any
from transformers import TrainerCallback


# --------------------------------------------------------------------------
# Core VRAM Measurement Functions
# --------------------------------------------------------------------------

def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get detailed memory information for a specific GPU.
    
    Args:
        device_id: GPU device ID (default: 0)
    
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_GB': 0.0,
            'reserved_GB': 0.0,
            'total_GB': 0.0,
            'free_GB': 0.0,
            'utilization_percent': 0.0
        }
    
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    free = total - reserved
    
    return {
        'allocated_GB': allocated / 1e9,
        'reserved_GB': reserved / 1e9,
        'total_GB': total / 1e9,
        'free_GB': free / 1e9,
        'utilization_percent': (reserved / total) * 100
    }


def get_total_vram_usage() -> Dict[str, float]:
    """
    Get total VRAM usage across all available GPUs.
    
    Returns:
        Dictionary with aggregated memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_GB': 0.0,
            'reserved_GB': 0.0,
            'total_available_GB': 0.0,
            'free_GB': 0.0,
            'utilization_percent': 0.0,
            'gpu_count': 0
        }
    
    total_allocated = 0
    total_reserved = 0
    total_available = 0
    gpu_count = torch.cuda.device_count()

    for i in range(gpu_count):
        total_allocated += torch.cuda.memory_allocated(i)
        total_reserved += torch.cuda.memory_reserved(i)
        total_available += torch.cuda.get_device_properties(i).total_memory

    total_free = total_available - total_reserved
    
    return {
        'allocated_GB': total_allocated / 1e9,
        'reserved_GB': total_reserved / 1e9,
        'total_available_GB': total_available / 1e9,
        'free_GB': total_free / 1e9,
        'utilization_percent': (total_reserved / total_available) * 100 if total_available > 0 else 0.0,
        'gpu_count': gpu_count
    }


def get_all_gpu_memory_info() -> List[Dict[str, float]]:
    """
    Get memory information for all available GPUs.
    
    Returns:
        List of dictionaries, one per GPU with memory statistics
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_info.append(get_gpu_memory_info(i))
    
    return gpu_info


# --------------------------------------------------------------------------
# VRAM Reporting Functions
# --------------------------------------------------------------------------

def print_vram_summary(show_per_gpu: bool = True, prefix: str = "🔥") -> None:
    """
    Print a comprehensive VRAM usage summary.
    
    Args:
        show_per_gpu: Whether to show per-GPU breakdown
        prefix: Emoji/prefix for the summary line
    """
    total_info = get_total_vram_usage()
    
    if total_info['gpu_count'] == 0:
        print(f"{prefix} No CUDA GPUs available")
        return
    
    print(f"{prefix} TOTAL VRAM: {total_info['allocated_GB']:.2f}GB allocated | "
          f"{total_info['reserved_GB']:.2f}GB reserved | "
          f"{total_info['utilization_percent']:.1f}% utilization "
          f"({total_info['gpu_count']} GPUs)")
    
    if show_per_gpu and total_info['gpu_count'] > 1:
        gpu_infos = get_all_gpu_memory_info()
        for i, gpu_info in enumerate(gpu_infos):
            print(f"   GPU {i}: {gpu_info['allocated_GB']:.2f}GB/"
                  f"{gpu_info['total_GB']:.2f}GB "
                  f"({gpu_info['utilization_percent']:.1f}%)")


def print_vram_compact() -> None:
    """Print a compact one-line VRAM summary."""
    total_info = get_total_vram_usage()
    if total_info['gpu_count'] == 0:
        print("💾 No CUDA GPUs")
    else:
        print(f"💾 VRAM: {total_info['reserved_GB']:.1f}GB/"
              f"{total_info['total_available_GB']:.1f}GB "
              f"({total_info['utilization_percent']:.0f}%)")


def print_vram_detailed() -> None:
    """Print detailed VRAM information for all GPUs."""
    total_info = get_total_vram_usage()
    
    if total_info['gpu_count'] == 0:
        print("❌ No CUDA GPUs available")
        return
    
    print("="*60)
    print("📊 DETAILED VRAM REPORT")
    print("="*60)
    
    gpu_infos = get_all_gpu_memory_info()
    for i, gpu_info in enumerate(gpu_infos):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        print(f"  Allocated: {gpu_info['allocated_GB']:.2f} GB")
        print(f"  Reserved:  {gpu_info['reserved_GB']:.2f} GB")
        print(f"  Free:      {gpu_info['free_GB']:.2f} GB")
        print(f"  Total:     {gpu_info['total_GB']:.2f} GB")
        print(f"  Usage:     {gpu_info['utilization_percent']:.1f}%")
        print()
    
    print(f"TOTAL ACROSS ALL GPUS:")
    print(f"  Allocated: {total_info['allocated_GB']:.2f} GB")
    print(f"  Reserved:  {total_info['reserved_GB']:.2f} GB")
    print(f"  Free:      {total_info['free_GB']:.2f} GB")
    print(f"  Total:     {total_info['total_available_GB']:.2f} GB")
    print(f"  Usage:     {total_info['utilization_percent']:.1f}%")
    print("="*60)


# --------------------------------------------------------------------------
# VRAM Monitoring Utilities
# --------------------------------------------------------------------------

class VRAMTracker:
    """Simple VRAM usage tracker for monitoring memory over time."""
    
    def __init__(self):
        self.snapshots = []
        self.labels = []
    
    def snapshot(self, label: str = "") -> Dict[str, float]:
        """Take a VRAM snapshot with optional label."""
        vram_info = get_total_vram_usage()
        vram_info['timestamp'] = time.time()
        vram_info['label'] = label
        
        self.snapshots.append(vram_info)
        self.labels.append(label)
        
        return vram_info
    
    def print_history(self) -> None:
        """Print all recorded snapshots."""
        if not self.snapshots:
            print("No VRAM snapshots recorded")
            return
        
        print("📈 VRAM History:")
        print("-" * 50)
        for i, snapshot in enumerate(self.snapshots):
            label = f" ({snapshot['label']})" if snapshot['label'] else ""
            print(f"{i+1:2d}. {snapshot['reserved_GB']:.2f}GB "
                  f"({snapshot['utilization_percent']:.1f}%){label}")
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get the snapshot with highest VRAM usage."""
        if not self.snapshots:
            return {}
        
        return max(self.snapshots, key=lambda x: x['reserved_GB'])
    
    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self.snapshots.clear()
        self.labels.clear()


# --------------------------------------------------------------------------
# Training Callbacks
# --------------------------------------------------------------------------

class VRAMMonitorCallback(TrainerCallback):
    """
    Comprehensive VRAM monitoring callback for Hugging Face trainers.
    Tracks memory usage at different training stages.
    """
    
    def __init__(self, 
                 detailed_logging: bool = False,
                 track_per_step: bool = False,
                 step_interval: int = 10):
        """
        Initialize VRAM monitoring callback.
        
        Args:
            detailed_logging: Print detailed VRAM info
            track_per_step: Track VRAM every N steps
            step_interval: Interval for step tracking
        """
        self.detailed_logging = detailed_logging
        self.track_per_step = track_per_step
        self.step_interval = step_interval
        self.epoch_start_time = None
        self.tracker = VRAMTracker()
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        print(f"\n{'='*60}")
        print("🚀 TRAINING STARTED - Initial VRAM State")
        print(f"{'='*60}")
        if self.detailed_logging:
            print_vram_detailed()
        else:
            print_vram_summary()
        self.tracker.snapshot("Training Start")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        epoch_num = int(state.epoch) + 1 if hasattr(state, 'epoch') else state.epoch
        print(f"\n{'='*60}")
        print(f"🚀 EPOCH {epoch_num} STARTED")
        print(f"{'='*60}")
        print_vram_summary()
        self.tracker.snapshot(f"Epoch {epoch_num} Start")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            epoch_num = int(state.epoch) if hasattr(state, 'epoch') else "Unknown"
            print(f"\n{'='*60}")
            print(f"✅ EPOCH {epoch_num} COMPLETED")
            print(f"⏱️  Duration: {epoch_time:.1f} seconds ({epoch_time/60:.1f} minutes)")
            print(f"{'='*60}")
            print_vram_summary()
            print(f"{'='*60}")
            self.tracker.snapshot(f"Epoch {epoch_num} End")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self.track_per_step and state.global_step % self.step_interval == 0:
            print(f"Step {state.global_step}: ", end="")
            print_vram_compact()
            self.tracker.snapshot(f"Step {state.global_step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        print(f"\n{'='*60}")
        print("🎉 TRAINING COMPLETED - Final VRAM State")
        print(f"{'='*60}")
        if self.detailed_logging:
            print_vram_detailed()
        else:
            print_vram_summary()
        self.tracker.snapshot("Training End")
        
        # Print peak usage
        peak = self.tracker.get_peak_usage()
        if peak:
            print(f"\n📊 Peak VRAM Usage: {peak['reserved_GB']:.2f}GB "
                  f"({peak['utilization_percent']:.1f}%) at {peak['label']}")


# --------------------------------------------------------------------------
# Convenience Functions
# --------------------------------------------------------------------------

def clear_vram_cache() -> None:
    """Clear CUDA cache and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_checkpoint(label: str = "", detailed: bool = False) -> Dict[str, float]:
    """
    Quick VRAM checkpoint with optional detailed output.
    
    Args:
        label: Optional label for the checkpoint
        detailed: Whether to print detailed information
    
    Returns:
        VRAM usage information
    """
    vram_info = get_total_vram_usage()
    
    if label:
        print(f"📍 {label}")
    
    if detailed:
        print_vram_detailed()
    else:
        print_vram_summary(show_per_gpu=False)
    
    return vram_info


# Example usage functions for different pipelines
def monitor_huggingface_training(trainer, detailed: bool = False) -> VRAMMonitorCallback:
    """
    Add VRAM monitoring to Hugging Face trainer.
    
    Args:
        trainer: Hugging Face trainer instance
        detailed: Whether to use detailed logging
    
    Returns:
        VRAMMonitorCallback instance
    """
    callback = VRAMMonitorCallback(detailed_logging=detailed)
    trainer.add_callback(callback)
    return callback


def monitor_model_loading(model_name: str) -> None:
    """Print VRAM state before and after model loading."""
    print(f"🔄 Loading {model_name}...")
    print("Before loading:")
    print_vram_summary()
    # User loads model here
    print("After loading:")
    print_vram_summary()