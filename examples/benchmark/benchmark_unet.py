import io
import os
import sys
import time

import fire
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusers import UNet2DConditionModel, ControlNetModel
from streamdiffusion.acceleration.tensorrt import compile_unet, compile_control_unet, UNet2DConditionControlNetModel
from streamdiffusion.acceleration.tensorrt.builder import create_onnx_path
from streamdiffusion.acceleration.tensorrt.engine import UNet2DConditionModelEngine, UNet2DConditionControlNetModelEngine
from streamdiffusion.acceleration.tensorrt.models import UNet, UNetWithControlNet


def run(
    iterations: int = 100,
    model_id_or_path: str = "stabilityai/sd-turbo",
    width: int = 512,
    height: int = 512,
    warmup: int = 10,
    steps: int = 3,
    use_controlnet: bool = False,
    num_controlnets: int = 1,
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
    steps : int, optional
        The number of denoising steps to run, by default 3.
    use_controlnet : bool, optional
        Whether to use ControlNet, by default False.
    num_controlnets : int, optional
        Number of ControlNet models to use (1 or 2), by default 1.
    """

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
        steps, 4, latent_height, latent_width, dtype=dtype, device=device
    )
    # Generate timesteps for the specified number of steps
    timestep = torch.randint(0, 1000, (steps,), dtype=torch.int64, device=device)
    encoder_hidden_states = torch.randn(
        steps, 77, 1024, dtype=dtype, device=device
    )

    print(f"Loading UNet model from {model_id_or_path}...")
    unet_torch = UNet2DConditionModel.from_pretrained(
        model_id_or_path, subfolder="unet", torch_dtype=dtype
    ).to(device)

    engine_dir = "tmp/engines"
    os.makedirs(engine_dir, exist_ok=True)
    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    if use_controlnet:
        if num_controlnets not in [1, 2]:
            raise ValueError("num_controlnets must be either 1 or 2")
            
        controlnet_models = [
            "thibaud/controlnet-sd21-depth-diffusers",
            "thibaud/controlnet-sd21-canny-diffusers"
        ]
        
        print(f"Loading {num_controlnets} ControlNet model(s)...")
        controlnets = []
        for i in range(num_controlnets):
            try:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_models[i], torch_dtype=dtype
                ).to(device)
                controlnets.append(controlnet)
            except Exception as e:
                print(f"Error loading {controlnet_models[i]}: {str(e)}")
                raise
        
        # Create dummy ControlNet inputs (shape: (num_controlnets, batch_size, channels, height, width))
        controlnet_images = torch.randn(
            num_controlnets, steps, 3, height, width, dtype=dtype, device=device
        )
        # Create scales with shape (num_controlnets, 1)
        controlnet_scales = torch.ones(num_controlnets, 1, dtype=dtype, device=device)
        
        # Create combined UNet+ControlNet model
        combined_model = UNet2DConditionControlNetModel(unet_torch, torch.nn.ModuleList(controlnets))
        
        # Use a unique engine path that includes both steps and num_controlnets
        unet_engine_path = f"{engine_dir}/controlnet_unet_steps_{steps}_num_controlnets_{num_controlnets}.engine"
        
        print("Compiling UNet+ControlNet TensorRT engine...")
        unet_model_data = UNetWithControlNet(
            fp16=True,
            device=device,
            num_controlnets=num_controlnets,
            max_batch_size=steps,
            min_batch_size=1,
            embedding_dim=unet_torch.config.cross_attention_dim,
            unet_dim=unet_torch.config.in_channels,
        )
        compile_control_unet(
            combined_model,
            unet_model_data,
            create_onnx_path("controlnet_unet", onnx_dir, opt=False),
            create_onnx_path("controlnet_unet", onnx_dir, opt=True),
            unet_engine_path,
            opt_batch_size=steps,
        )
        del combined_model, unet_torch, controlnets
        torch.cuda.empty_cache()

        print("Loading UNet+ControlNet TensorRT engine...")
        import polygraphy.cuda
        cuda_stream = polygraphy.cuda.Stream()
        unet_trt = UNet2DConditionControlNetModelEngine(unet_engine_path, cuda_stream)

        print("Warmup...")
        for _ in tqdm(range(warmup)):
            _ = unet_trt(
                latent_model_input,
                timestep,
                encoder_hidden_states,
                controlnet_images,
                controlnet_scales,
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
                controlnet_images,
                controlnet_scales,
            )
            end_event.record()
            torch.cuda.synchronize()
            results.append(start_event.elapsed_time(end_event))

    else:
        unet_engine_path = f"{engine_dir}/unet_steps_{steps}.engine"

        print("Compiling UNet TensorRT engine...")
        unet_model_data = UNet(
            fp16=True,
            device=device,
            max_batch_size=steps, # Max batch size for dummy inputs
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
            opt_batch_size=steps,
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

    print(f"\nBenchmark Results ({'ControlNet + UNet' if use_controlnet else 'UNet Only'}):")
    print(f"Average latency: {sum(results) / len(results):.2f}ms")
    import numpy as np

    latency_arr = np.array(results)
    print(f"Min latency: {np.min(latency_arr):.2f}ms")
    print(f"Max latency: {np.max(latency_arr):.2f}ms")
    print(f"Std latency: {np.std(latency_arr):.2f}ms")


if __name__ == "__main__":
    fire.Fire(run) 