import io
import os
import sys
import time

import fire
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusers import UNet2DConditionModel
from streamdiffusion.acceleration.tensorrt import compile_unet
from streamdiffusion.acceleration.tensorrt.builder import create_onnx_path
from streamdiffusion.acceleration.tensorrt.engine import UNet2DConditionModelEngine
from streamdiffusion.acceleration.tensorrt.models import UNet


def run(
    iterations: int = 100,
    model_id_or_path: str = "stabilityai/sd-turbo",
    width: int = 512,
    height: int = 512,
    warmup: int = 10,
):
    """
    Benchmarks the UNet forward pass using TensorRT.

    Parameters
    ----------
    iterations : int, optional
        The number of iterations to run, by default 100.
    model_id_or_path : str
        The model id or path to load.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    warmup : int, optional
        The number of warmup steps to perform, by default 10.
    """

    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda"
    dtype = torch.float16

    # Dummy inputs based on observed shapes and dtypes in single.py
    # x_t_latent_plus_uc: torch.Size([3, 4, 64, 64]) torch.float16
    # t_list: torch.Size([3]) torch.int64
    # self.prompt_embeds: torch.Size([3, 77, 1024]) torch.float16

    latent_height = int(height // 8)
    latent_width = int(width // 8)

    latent_model_input = torch.randn(
        3, 4, latent_height, latent_width, dtype=dtype, device=device
    )
    timestep = torch.tensor([32, 32, 45], dtype=torch.int64, device=device) # Example timesteps
    encoder_hidden_states = torch.randn(
        3, 77, 1024, dtype=dtype, device=device
    )

    print(f"Loading UNet model from {model_id_or_path}...")
    unet_torch = UNet2DConditionModel.from_pretrained(
        model_id_or_path, subfolder="unet", torch_dtype=dtype
    ).to(device)

    engine_dir = "tmp/engines"
    os.makedirs(engine_dir, exist_ok=True)
    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"

    print("Compiling UNet TensorRT engine...")
    unet_model_data = UNet(
        fp16=True,
        device=device,
        max_batch_size=3, # Max batch size for dummy inputs
        min_batch_size=1, # Min batch size for dummy inputs
        embedding_dim=unet_torch.config.cross_attention_dim, # Typically 768
        unet_dim=unet_torch.config.in_channels, # Typically 4
    )
    compile_unet(
        unet_torch,
        unet_model_data,
        create_onnx_path("unet", onnx_dir, opt=False),
        create_onnx_path("unet", onnx_dir, opt=True),
        unet_engine_path,
        opt_batch_size=3,
    )
    del unet_torch
    torch.cuda.empty_cache()

    print("Loading UNet TensorRT engine...")
    import polygraphy.cuda
    cuda_stream = polygraphy.cuda.Stream()
    unet_trt = UNet2DConditionModelEngine(unet_engine_path, cuda_stream)

    print("Warmup...")
    for _ in tqdm(range(warmup)):
        _ = unet_trt(
            latent_model_input,
            timestep,
            encoder_hidden_states,
        )
    
    results = []
    print("Benchmarking...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in tqdm(range(iterations)):
        start_event.record()
        _ = unet_trt(
            latent_model_input,
            timestep,
            encoder_hidden_states,
        )
        end_event.record()
        torch.cuda.synchronize()
        results.append(start_event.elapsed_time(end_event))

    print(f"\nAverage time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")
    import numpy as np

    fps_arr = 1000 / np.array(results)
    print(f"Max FPS: {np.max(fps_arr)}")
    print(f"Min FPS: {np.min(fps_arr)}")
    print(f"Std: {np.std(fps_arr)}")


if __name__ == "__main__":
    fire.Fire(run) 