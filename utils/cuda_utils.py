import torch


def set_cuda():
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
