import torch


def print_cuda_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**2
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**2
        free_memory_approx = total_memory - reserved_memory

        memory_info = (
            f"CUDA Memory (in MB)\n"
            f"Total: {total_memory:.2f} MB\n"
            f"Allocated: {allocated_memory:.2f} MB\n"
            f"Reserved: {reserved_memory:.2f} MB\n"
            f"Free (approx.): {free_memory_approx:.2f} MB"
        )

        return memory_info
    else:
        return "CUDA not available"
