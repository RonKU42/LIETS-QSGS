"""Shared utility helpers."""

import psutil


def print_system_mem(tag: str = '') -> None:
    """Print a short system RAM summary."""
    vm = psutil.virtual_memory()
    used = vm.used / 1024 ** 3
    avail = vm.available / 1024 ** 3
    total = vm.total / 1024 ** 3
    print(
        f"{tag} System RAM: used={used:.2f} GB, "
        f"available={avail:.2f} GB, total={total:.2f} GB, "
        f"percent={vm.percent:.1f}%"
    )
