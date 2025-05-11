import torch

def print_model_and_gpu_stats(model, device=torch.device('cuda:0')):
    # 1) Estimate model size on GPU (parameters only, in MB)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb    = param_bytes / 1024**2
    print(f"Model parameters: {param_mb:.2f} MB")

    if device.type == 'cuda':
        # make sure we're measuring from the right device
        torch.cuda.synchronize(device)

        # 2) GPU memory stats (in MB)
        allocated_mb = torch.cuda.memory_allocated(device)  / 1024**2
        reserved_mb  = torch.cuda.memory_reserved(device)   / 1024**2
        total_mb     = torch.cuda.get_device_properties(device).total_memory / 1024**2
        free_mb      = total_mb - reserved_mb

        print(f"CUDA total memory   : {total_mb:7.2f} MB")
        print(f"CUDA allocated      : {allocated_mb:7.2f} MB")
        print(f"CUDA reserved       : {reserved_mb:7.2f} MB")
        print(f"CUDA free (est.)    : {free_mb:7.2f} MB")
    else:
        print("Device is not CUDA, skipping GPU stats.")
        