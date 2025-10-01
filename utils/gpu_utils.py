import os
import subprocess


def setup_gpu(min_free_gb=10):
    # Query free memory (MiB) for each GPU
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"
    ]).decode().strip().splitlines()
    free_mib = [int(x) for x in out]
    # Choose the GPU with enough free memory, otherwise take the freest one
    candidates = [i for i, mib in enumerate(free_mib) if mib >= min_free_gb * 1024]
    gpu = candidates[0] if candidates else max(range(len(free_mib)), key=lambda i: free_mib[i])

    # Make only that GPU visible to JAX
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Make JAX allocate on demand instead of preallocating ~90% of VRAM
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # (Optional) cap how much JAX may use (e.g., 90%)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

    print(f"Picked GPU {gpu} (free MiB per GPU: {free_mib})")

    import jax
    print("JAX sees devices:", jax.devices())
    import tensorflow as tf
    # job performed on GPU by default
    device_name = tf.test.gpu_device_name()
    if "GPU" not in device_name:
        print("TF: GPU not found")
    else:
        print('TF: Found GPU at: {}'.format(device_name))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # same on all
