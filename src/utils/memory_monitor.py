"""Memory monitoring utility for development and production."""
import gc
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed. Memory monitoring disabled.")


def get_memory_usage() -> Dict:
    """
    Get current memory usage of the process.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    if not PSUTIL_AVAILABLE:
        return {
            "available": False,
            "message": "psutil not installed"
        }
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Get system memory info
    virtual_mem = psutil.virtual_memory()
    
    return {
        "available": True,
        "process_mb": round(memory_info.rss / 1024 / 1024, 2),
        "process_percent": round(process.memory_percent(), 2),
        "system_total_mb": round(virtual_mem.total / 1024 / 1024, 2),
        "system_available_mb": round(virtual_mem.available / 1024 / 1024, 2),
        "system_percent": virtual_mem.percent
    }


def log_memory_usage(label: str = "Memory"):
    """
    Log current memory usage with a label.
    
    Args:
        label: Label for the log message
    """
    stats = get_memory_usage()
    if stats["available"]:
        logger.info(
            f"{label}: {stats['process_mb']:.1f}MB "
            f"({stats['process_percent']:.1f}% of system)"
        )
    else:
        logger.debug(f"{label}: Memory monitoring not available")


def check_memory_limit(limit_mb: int = 512, warning_threshold: float = 0.8) -> bool:
    """
    Check if memory usage is approaching the limit.
    
    Args:
        limit_mb: Memory limit in MB (default: 512 for Render free tier)
        warning_threshold: Threshold for warning (default: 0.8 = 80%)
        
    Returns:
        True if memory is OK, False if approaching or exceeding limit
    """
    stats = get_memory_usage()
    if not stats["available"]:
        return True
    
    current_mb = stats["process_mb"]
    usage_ratio = current_mb / limit_mb
    
    if usage_ratio >= 1.0:
        logger.error(
            f"⚠️ MEMORY LIMIT EXCEEDED: {current_mb:.1f}MB / {limit_mb}MB "
            f"({usage_ratio*100:.1f}%)"
        )
        return False
    elif usage_ratio >= warning_threshold:
        logger.warning(
            f"⚠️ Memory approaching limit: {current_mb:.1f}MB / {limit_mb}MB "
            f"({usage_ratio*100:.1f}%)"
        )
        # Try to force garbage collection
        gc.collect()
        return False
    else:
        logger.info(
            f"✅ Memory OK: {current_mb:.1f}MB / {limit_mb}MB "
            f"({usage_ratio*100:.1f}%)"
        )
        return True


def format_memory_stats() -> str:
    """
    Format memory statistics as a human-readable string.
    
    Returns:
        Formatted string with memory stats
    """
    stats = get_memory_usage()
    if not stats["available"]:
        return "Memory monitoring not available (install psutil)"
    
    return (
        f"Process: {stats['process_mb']}MB ({stats['process_percent']}%)\n"
        f"System: {stats['system_available_mb']:.0f}MB available / "
        f"{stats['system_total_mb']:.0f}MB total ({stats['system_percent']}% used)"
    )


if __name__ == "__main__":
    # Test memory monitoring
    print("Memory Monitoring Test")
    print("=" * 50)
    print(format_memory_stats())
    print()
    
    # Check Render free tier limit
    print("Checking Render free tier limit (512MB):")
    check_memory_limit(limit_mb=512)
