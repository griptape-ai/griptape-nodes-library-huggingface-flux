"""
GPU discovery and management utilities for FLUX CUDA nodes.
Cross-platform support for Windows and Linux.
"""

import logging
from typing import List, Dict, Optional, Tuple
import platform
import subprocess
import psutil

logger = logging.getLogger(__name__)


def _get_torch():
    """Get torch from shared backend"""
    try:
        from .. import get_shared_backend
        backend = get_shared_backend()
        if backend and backend.get('available', False):
            return backend['torch']
    except:
        pass
    
    # Fallback to direct import if shared backend not available
    try:
        import torch
        return torch
    except ImportError:
        logger.error("Torch not available in shared backend or direct import")
        return None


class GPUManager:
    """Manages GPU discovery, selection, and memory allocation."""
    
    def __init__(self):
        self.torch = _get_torch()
        self.available_devices = self._discover_gpus()
        self.current_device = None
        self.allocated_memory = {}
        
    def _discover_gpus(self) -> List[Dict]:
        """Discover available CUDA GPUs with detailed information."""
        devices = []
        
        if not self.torch or not self.torch.cuda.is_available():
            logger.warning("CUDA is not available on this system")
            return devices
            
        try:
            device_count = self.torch.cuda.device_count()
            
            for i in range(device_count):
                device_info = self._get_device_info(i)
                devices.append(device_info)
                logger.info(f"Discovered GPU {i}: {device_info['name']} ({device_info['memory_gb']:.1f}GB)")
                
        except Exception as e:
            logger.error(f"Error discovering GPUs: {e}")
            
        return devices
    
    def _get_device_info(self, device_id: int) -> Dict:
        """Get detailed information about a specific GPU device."""
        try:
            # Get basic PyTorch info
            props = self.torch.cuda.get_device_properties(device_id)
            memory_gb = props.total_memory / (1024**3)
            
            # Get current memory usage
            self.torch.cuda.set_device(device_id)
            memory_allocated = self.torch.cuda.memory_allocated(device_id) / (1024**3)
            memory_reserved = self.torch.cuda.memory_reserved(device_id) / (1024**3)
            memory_free = memory_gb - memory_reserved
            
            device_info = {
                'id': device_id,
                'name': props.name,
                'memory_gb': memory_gb,
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved, 
                'memory_free_gb': memory_free,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': getattr(props, 'multiprocessor_count', getattr(props, 'multi_processor_count', 0)),
                'is_available': True
            }
            
            # Add platform-specific info
            if platform.system() == "Linux":
                device_info.update(self._get_linux_gpu_info(device_id))
            elif platform.system() == "Windows":
                device_info.update(self._get_windows_gpu_info(device_id))
                
            return device_info
            
        except Exception as e:
            logger.error(f"Error getting info for GPU {device_id}: {e}")
            return {
                'id': device_id,
                'name': f"GPU {device_id}",
                'memory_gb': 0.0,
                'memory_allocated_gb': 0.0,
                'memory_reserved_gb': 0.0,
                'memory_free_gb': 0.0,
                'compute_capability': "Unknown",
                'multiprocessor_count': 0,
                'is_available': False,
                'error': str(e)
            }
    
    def _get_linux_gpu_info(self, device_id: int) -> Dict:
        """Get Linux-specific GPU information using nvidia-smi."""
        info = {}
        try:
            # Try to get temperature and power usage
            cmd = [
                'nvidia-smi', 
                '--query-gpu=temperature.gpu,power.draw,utilization.gpu',
                '--format=csv,noheader,nounits',
                f'--id={device_id}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 3:
                    info.update({
                        'temperature_c': float(values[0]) if values[0] != '[Not Supported]' else None,
                        'power_draw_w': float(values[1]) if values[1] != '[Not Supported]' else None,
                        'utilization_percent': float(values[2]) if values[2] != '[Not Supported]' else None
                    })
                    
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            logger.debug(f"Could not get nvidia-smi info for GPU {device_id}: {e}")
            
        return info
    
    def _get_windows_gpu_info(self, device_id: int) -> Dict:
        """Get Windows-specific GPU information."""
        info = {}
        try:
            # Try nvidia-smi on Windows (usually in PATH if drivers installed)
            cmd = [
                'nvidia-smi', 
                '--query-gpu=temperature.gpu,power.draw,utilization.gpu',
                '--format=csv,noheader,nounits',
                f'--id={device_id}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 3:
                    info.update({
                        'temperature_c': float(values[0]) if values[0] != '[Not Supported]' else None,
                        'power_draw_w': float(values[1]) if values[1] != '[Not Supported]' else None,
                        'utilization_percent': float(values[2]) if values[2] != '[Not Supported]' else None
                    })
                    
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            logger.debug(f"Could not get nvidia-smi info for GPU {device_id}: {e}")
            
        return info
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available GPU devices with current status."""
        # Refresh memory info
        for device in self.available_devices:
            if device['is_available']:
                device_id = device['id']
                try:
                    self.torch.cuda.set_device(device_id)
                    device['memory_allocated_gb'] = self.torch.cuda.memory_allocated(device_id) / (1024**3)
                    device['memory_reserved_gb'] = self.torch.cuda.memory_reserved(device_id) / (1024**3)
                    device['memory_free_gb'] = device['memory_gb'] - device['memory_reserved_gb']
                except Exception as e:
                    logger.debug(f"Could not update memory info for GPU {device_id}: {e}")
                    
        return self.available_devices
    
    def select_device(self, device_id: int, memory_fraction: float = 0.8) -> bool:
        """
        Select and configure a GPU device with memory allocation.
        
        Args:
            device_id: GPU device ID to select
            memory_fraction: Fraction of GPU memory to allocate (0.1-1.0)
            
        Returns:
            bool: True if device was successfully selected and configured
        """
        try:
            # Validate device_id
            if device_id >= len(self.available_devices):
                logger.error(f"Device ID {device_id} not available. Available devices: {len(self.available_devices)}")
                return False
                
            device_info = self.available_devices[device_id]
            if not device_info['is_available']:
                logger.error(f"Device {device_id} is not available: {device_info.get('error', 'Unknown error')}")
                return False
            
            # Validate memory fraction
            if not (0.1 <= memory_fraction <= 1.0):
                logger.error(f"Memory fraction {memory_fraction} must be between 0.1 and 1.0")
                return False
            
            # Set device
            self.torch.cuda.set_device(device_id)
            
            # Set memory fraction
            self.torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)
            
            # Store current configuration
            self.current_device = device_id
            self.allocated_memory[device_id] = memory_fraction
            
            logger.info(f"Selected GPU {device_id}: {device_info['name']} "
                       f"with {memory_fraction*100:.0f}% memory allocation "
                       f"({memory_fraction * device_info['memory_gb']:.1f}GB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error selecting device {device_id}: {e}")
            return False
    
    def get_device_memory_info(self, device_id: Optional[int] = None) -> Dict:
        """Get detailed memory information for a device."""
        if device_id is None:
            device_id = self.current_device or 0
            
        try:
            self.torch.cuda.set_device(device_id)
            
            props = self.torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory / (1024**3)
            allocated_memory = self.torch.cuda.memory_allocated(device_id) / (1024**3)
            reserved_memory = self.torch.cuda.memory_reserved(device_id) / (1024**3)
            free_memory = total_memory - reserved_memory
            
            memory_fraction = self.allocated_memory.get(device_id, 1.0)
            allocated_limit = total_memory * memory_fraction
            
            return {
                'device_id': device_id,
                'total_memory_gb': total_memory,
                'allocated_memory_gb': allocated_memory,
                'reserved_memory_gb': reserved_memory,
                'free_memory_gb': free_memory,
                'memory_fraction': memory_fraction,
                'allocated_limit_gb': allocated_limit,
                'available_for_allocation_gb': allocated_limit - allocated_memory
            }
            
        except Exception as e:
            logger.error(f"Error getting memory info for device {device_id}: {e}")
            return {}
    
    def cleanup_device_memory(self, device_id: Optional[int] = None):
        """Clean up GPU memory cache."""
        if device_id is None:
            device_id = self.current_device or 0
            
        try:
            self.torch.cuda.set_device(device_id)
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize()
            logger.info(f"Cleaned up memory cache for GPU {device_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up memory for device {device_id}: {e}")
    
    def get_optimal_device_for_model(self, required_memory_gb: float) -> Optional[int]:
        """
        Find the optimal GPU device for a model requiring specific memory.
        
        Args:
            required_memory_gb: Required memory in GB
            
        Returns:
            Optional device ID that can fit the model, or None if none available
        """
        suitable_devices = []
        
        for device in self.available_devices:
            if not device['is_available']:
                continue
                
            # Consider memory fraction if device is already allocated
            device_id = device['id']
            memory_fraction = self.allocated_memory.get(device_id, 0.8)  # Default 80%
            available_memory = device['memory_gb'] * memory_fraction
            
            if available_memory >= required_memory_gb:
                # Prefer device with least utilization and most free memory
                score = device['memory_free_gb'] - required_memory_gb
                suitable_devices.append((device_id, score, device))
        
        if not suitable_devices:
            logger.warning(f"No suitable device found for {required_memory_gb:.1f}GB requirement")
            return None
            
        # Sort by score (most free memory after allocation)
        suitable_devices.sort(key=lambda x: x[1], reverse=True)
        best_device = suitable_devices[0]
        
        logger.info(f"Optimal device for {required_memory_gb:.1f}GB: "
                   f"GPU {best_device[0]} ({best_device[2]['name']}) "
                   f"with {best_device[1]:.1f}GB free after allocation")
        
        return best_device[0]


# Global GPU manager instance
gpu_manager = GPUManager()


def get_available_gpus() -> List[Dict]:
    """Get list of available GPUs. Convenience function."""
    return gpu_manager.get_available_devices()


def select_gpu(device_id: int, memory_fraction: float = 0.8) -> bool:
    """Select GPU device with memory allocation. Convenience function."""
    return gpu_manager.select_device(device_id, memory_fraction)


def get_gpu_memory_info(device_id: Optional[int] = None) -> Dict:
    """Get GPU memory information. Convenience function.""" 
    return gpu_manager.get_device_memory_info(device_id)


def cleanup_gpu_memory(device_id: Optional[int] = None):
    """Clean up GPU memory. Convenience function."""
    gpu_manager.cleanup_device_memory(device_id)


def find_optimal_gpu(required_memory_gb: float) -> Optional[int]:
    """Find optimal GPU for required memory. Convenience function."""
    return gpu_manager.get_optimal_device_for_model(required_memory_gb)


if __name__ == "__main__":
    # Test GPU discovery
    print("=== GPU Discovery Test ===")
    devices = get_available_gpus()
    
    if not devices:
        print("No CUDA GPUs found")
    else:
        for device in devices:
            print(f"\nGPU {device['id']}: {device['name']}")
            print(f"  Memory: {device['memory_gb']:.1f}GB total, {device['memory_free_gb']:.1f}GB free")
            print(f"  Compute: {device['compute_capability']}")
            if 'temperature_c' in device and device['temperature_c'] is not None:
                print(f"  Temperature: {device['temperature_c']}°C")
            if 'utilization_percent' in device and device['utilization_percent'] is not None:
                print(f"  Utilization: {device['utilization_percent']}%")
        
        # Test device selection
        print(f"\n=== Testing Device Selection ===")
        if select_gpu(0, 0.8):
            memory_info = get_gpu_memory_info(0)
            print(f"Selected GPU 0 with 80% memory allocation")
            print(f"Allocated limit: {memory_info['allocated_limit_gb']:.1f}GB")
            print(f"Available for allocation: {memory_info['available_for_allocation_gb']:.1f}GB")
        
        # Test optimal device selection
        print(f"\n=== Testing Optimal Device Selection ===")
        optimal_device = find_optimal_gpu(8.0)  # 8GB requirement
        if optimal_device is not None:
            print(f"Optimal device for 8GB model: GPU {optimal_device}")
        else:
            print("No device suitable for 8GB model") 