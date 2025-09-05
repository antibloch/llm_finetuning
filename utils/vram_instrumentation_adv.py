"""
Enhanced VRAM Instrumentation Module
===================================
Adds granular memory tracking for model, gradients, and optimizer states
"""

import torch
import time
import gc
from typing import Dict, List, Optional, Any, Tuple
from transformers import TrainerCallback


# --------------------------------------------------------------------------
# Enhanced VRAM Measurement Functions
# --------------------------------------------------------------------------

def get_model_memory_breakdown(model) -> Dict[str, float]:
    """
    Get detailed memory breakdown for model parameters.
    
    Args:
        model: PyTorch model (can be DDP wrapped)
    
    Returns:
        Dictionary with memory breakdown in GB
    """
    if not torch.cuda.is_available():
        return {
            'model_params_GB': 0.0,
            'model_buffers_GB': 0.0,
            'total_model_GB': 0.0,
            'param_count': 0,
            'buffer_count': 0
        }
    
    # Handle DDP wrapped models
    actual_model = model.module if hasattr(model, 'module') else model
    
    param_memory = 0
    buffer_memory = 0
    param_count = 0
    buffer_count = 0
    
    # Calculate parameter memory
    for param in actual_model.parameters():
        if param.is_cuda:
            param_memory += param.numel() * param.element_size()
            param_count += param.numel()
    
    # Calculate buffer memory (non-parameter tensors like batch norm running stats)
    for buffer in actual_model.buffers():
        if buffer.is_cuda:
            buffer_memory += buffer.numel() * buffer.element_size()
            buffer_count += buffer.numel()
    
    return {
        'model_params_GB': param_memory / 1e9,
        'model_buffers_GB': buffer_memory / 1e9,
        'total_model_GB': (param_memory + buffer_memory) / 1e9,
        'param_count': param_count,
        'buffer_count': buffer_count
    }


def get_gradient_memory_breakdown(model) -> Dict[str, float]:
    """
    Get memory usage of gradients.
    
    Args:
        model: PyTorch model (can be DDP wrapped)
    
    Returns:
        Dictionary with gradient memory breakdown in GB
    """
    if not torch.cuda.is_available():
        return {
            'gradient_memory_GB': 0.0,
            'params_with_grad': 0,
            'total_params': 0
        }
    
    # Handle DDP wrapped models
    actual_model = model.module if hasattr(model, 'module') else model
    
    gradient_memory = 0
    params_with_grad = 0
    total_params = 0
    
    for param in actual_model.parameters():
        total_params += 1
        if param.grad is not None and param.grad.is_cuda:
            gradient_memory += param.grad.numel() * param.grad.element_size()
            params_with_grad += 1
    
    return {
        'gradient_memory_GB': gradient_memory / 1e9,
        'params_with_grad': params_with_grad,
        'total_params': total_params
    }


def get_optimizer_memory_breakdown(optimizer) -> Dict[str, float]:
    """
    Get memory usage of optimizer states.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Dictionary with optimizer memory breakdown in GB
    """
    if not torch.cuda.is_available():
        return {
            'optimizer_state_GB': 0.0,
            'state_entries': 0
        }
    
    optimizer_memory = 0
    state_entries = 0
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                state = optimizer.state[param]
                state_entries += 1
                
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        optimizer_memory += value.numel() * value.element_size()
    
    return {
        'optimizer_state_GB': optimizer_memory / 1e9,
        'state_entries': state_entries
    }


def get_comprehensive_memory_breakdown(model, optimizer=None) -> Dict[str, Any]:
    """
    Get comprehensive memory breakdown including model, gradients, optimizer, and total VRAM.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
    
    Returns:
        Dictionary with complete memory breakdown
    """
    model_breakdown = get_model_memory_breakdown(model)
    gradient_breakdown = get_gradient_memory_breakdown(model)
    optimizer_breakdown = get_optimizer_memory_breakdown(optimizer) if optimizer else {
        'optimizer_state_GB': 0.0, 'state_entries': 0
    }
    vram_info = get_total_vram_usage()
    
    # Calculate known vs unknown memory
    known_memory = (model_breakdown['total_model_GB'] + 
                   gradient_breakdown['gradient_memory_GB'] + 
                   optimizer_breakdown['optimizer_state_GB'])
    
    unknown_memory = vram_info['allocated_GB'] - known_memory
    
    return {
        'model': model_breakdown,
        'gradients': gradient_breakdown,
        'optimizer': optimizer_breakdown,
        'vram_total': vram_info,
        'summary': {
            'known_memory_GB': known_memory,
            'unknown_memory_GB': max(0, unknown_memory),  # Don't go negative
            'total_allocated_GB': vram_info['allocated_GB'],
            'memory_efficiency_percent': (known_memory / vram_info['allocated_GB'] * 100) if vram_info['allocated_GB'] > 0 else 0
        }
    }


def print_comprehensive_memory_report(model, optimizer=None, device_id: int = 0) -> None:
    """
    Print a detailed memory report showing breakdown by component.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        device_id: CUDA device ID
    """
    breakdown = get_comprehensive_memory_breakdown(model, optimizer)
    
    print("=" * 80)
    print("📊 COMPREHENSIVE MEMORY BREAKDOWN")
    print("=" * 80)
    
    # Model memory
    model_info = breakdown['model']
    print(f"🧠 MODEL MEMORY:")
    print(f"   Parameters: {model_info['model_params_GB']:.2f} GB ({model_info['param_count']:,} params)")
    print(f"   Buffers:    {model_info['model_buffers_GB']:.2f} GB ({model_info['buffer_count']:,} elements)")
    print(f"   Total:      {model_info['total_model_GB']:.2f} GB")
    
    # Gradient memory
    grad_info = breakdown['gradients']
    print(f"\n📈 GRADIENT MEMORY:")
    print(f"   Gradients:  {grad_info['gradient_memory_GB']:.2f} GB")
    print(f"   Coverage:   {grad_info['params_with_grad']}/{grad_info['total_params']} parameters have gradients")
    
    # Optimizer memory
    opt_info = breakdown['optimizer']
    print(f"\n⚡ OPTIMIZER MEMORY:")
    print(f"   States:     {opt_info['optimizer_state_GB']:.2f} GB")
    print(f"   Entries:    {opt_info['state_entries']} state entries")
    
    # Summary
    summary = breakdown['summary']
    vram_total = breakdown['vram_total']
    print(f"\n📋 MEMORY SUMMARY:")
    print(f"   Known Memory:     {summary['known_memory_GB']:.2f} GB")
    print(f"   Unknown Memory:   {summary['unknown_memory_GB']:.2f} GB (activations, cache, etc.)")
    print(f"   Total Allocated:  {summary['total_allocated_GB']:.2f} GB")
    print(f"   Total Available:  {vram_total['total_available_GB']:.2f} GB")
    print(f"   Utilization:      {vram_total['utilization_percent']:.1f}%")
    print(f"   Efficiency:       {summary['memory_efficiency_percent']:.1f}% (known/allocated)")
    
    print("=" * 80)


def get_per_gpu_memory_breakdown(model, optimizer=None) -> List[Dict[str, Any]]:
    """
    Get memory breakdown for each GPU separately.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
    
    Returns:
        List of memory breakdowns, one per GPU
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_breakdowns = []
    
    for device_id in range(torch.cuda.device_count()):
        # Get basic VRAM info for this GPU
        gpu_vram = get_gpu_memory_info(device_id)
        
        # For now, we can't easily split model/optimizer memory per GPU
        # This would require more complex tracking during model placement
        gpu_breakdown = {
            'device_id': device_id,
            'device_name': torch.cuda.get_device_name(device_id),
            'vram': gpu_vram,
            'note': 'Per-GPU model/optimizer breakdown requires device-specific tracking'
        }
        
        gpu_breakdowns.append(gpu_breakdown)
    
    return gpu_breakdowns


# --------------------------------------------------------------------------
# Enhanced Training Monitoring
# --------------------------------------------------------------------------

class EnhancedVRAMTracker:
    """Enhanced VRAM tracker with component-wise memory tracking."""
    
    def __init__(self):
        self.snapshots = []
        self.component_history = []
    
    def snapshot_comprehensive(self, model, optimizer=None, label: str = "") -> Dict[str, Any]:
        """Take a comprehensive snapshot including component breakdown."""
        breakdown = get_comprehensive_memory_breakdown(model, optimizer)
        breakdown['timestamp'] = time.time()
        breakdown['label'] = label
        
        self.snapshots.append(breakdown)
        return breakdown
    
    def print_component_history(self) -> None:
        """Print history of component memory usage."""
        if not self.snapshots:
            print("No comprehensive snapshots recorded")
            return
        
        print("📈 COMPONENT MEMORY HISTORY:")
        print("-" * 100)
        print(f"{'#':<3} {'Label':<25} {'Model':<8} {'Grad':<8} {'Opt':<8} {'Total':<8} {'VRAM%':<8}")
        print("-" * 100)
        
        for i, snapshot in enumerate(self.snapshots):
            model_gb = snapshot['model']['total_model_GB']
            grad_gb = snapshot['gradients']['gradient_memory_GB']
            opt_gb = snapshot['optimizer']['optimizer_state_GB']
            total_gb = snapshot['summary']['known_memory_GB']
            vram_percent = snapshot['vram_total']['utilization_percent']
            label = snapshot['label'][:24]
            
            print(f"{i+1:<3} {label:<25} {model_gb:<8.2f} {grad_gb:<8.2f} {opt_gb:<8.2f} {total_gb:<8.2f} {vram_percent:<8.1f}")
    
    def get_peak_component_usage(self) -> Dict[str, Any]:
        """Get snapshot with highest total known memory usage."""
        if not self.snapshots:
            return {}
        
        return max(self.snapshots, key=lambda x: x['summary']['known_memory_GB'])
    
    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self.snapshots.clear()
        self.component_history.clear()


# --------------------------------------------------------------------------
# Integration with Existing Functions
# --------------------------------------------------------------------------

def enhanced_vram_checkpoint(model, optimizer=None, label: str = "", detailed: bool = False) -> Dict[str, Any]:
    """
    Enhanced VRAM checkpoint with component breakdown.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        label: Optional label for the checkpoint
        detailed: Whether to print detailed breakdown
    
    Returns:
        Comprehensive memory breakdown
    """
    if label:
        print(f"📍 {label}")
    
    if detailed:
        print_comprehensive_memory_report(model, optimizer)
    else:
        breakdown = get_comprehensive_memory_breakdown(model, optimizer)
        summary = breakdown['summary']
        vram = breakdown['vram_total']
        print(f"💾 Memory: Model {breakdown['model']['total_model_GB']:.1f}GB + "
              f"Grad {breakdown['gradients']['gradient_memory_GB']:.1f}GB + "
              f"Opt {breakdown['optimizer']['optimizer_state_GB']:.1f}GB = "
              f"{summary['known_memory_GB']:.1f}GB known "
              f"({vram['utilization_percent']:.0f}% VRAM)")
    
    return get_comprehensive_memory_breakdown(model, optimizer)


# Import existing functions to maintain compatibility
from utils.vram_instrumentation import (
    get_gpu_memory_info,
    get_total_vram_usage,
    get_all_gpu_memory_info,
    print_vram_summary,
    print_vram_compact,
    print_vram_detailed,
    VRAMTracker,
    VRAMMonitorCallback,
    clear_vram_cache,
    vram_checkpoint
)