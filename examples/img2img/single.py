import os
import sys
import time
from typing import Literal, Dict, Optional, List

import fire
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "chillguy.png"),
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.png"),
    model_id_or_path: str = "stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "elon musk",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 2,
    delta: float = 0.5,
    use_controlnet: bool = True,
    controlnet_model_ids: Optional[List[str]] = ["thibaud/controlnet-sd21-depth-diffusers"],
    controlnet_scales: Optional[List[float]] = [0.5],
    controlnet_image_paths: Optional[List[str]] = None,
):
    """
    Initializes the StreamDiffusionWrapper.

    Parameters
    ----------
    input : str, optional
        The input image file to load images from.
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The model id or path to load.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    use_controlnet : bool, optional
        Whether to use ControlNet or not, by default False.
    controlnet_model_ids : Optional[List[str]], optional
        List of ControlNet model IDs to load, by default None.
        Example: ["thibaud/controlnet-sd21-depth-diffusers", "thibaud/controlnet-sd21-canny-diffusers"]
    controlnet_scales : Optional[List[float]], optional
        List of ControlNet conditioning scales, by default None.
        If None, defaults to [1.0] for each ControlNet.
    controlnet_image_paths : Optional[List[str]], optional
        List of paths to ControlNet conditioning images, by default None.
        If None and use_controlnet is True, will use default depth image.
        Example: ["path/to/depth.png", "path/to/canny.png"]
    """

    if guidance_scale <= 1.0:
        cfg_type = "none"

    # Set default ControlNet models and image paths if enabled but not specified
    if use_controlnet:
        if controlnet_model_ids is None:
            controlnet_model_ids = ["thibaud/controlnet-sd21-depth-diffusers"]
            print(f"Using default ControlNet model: {controlnet_model_ids}")
        
        if controlnet_image_paths is None:
            # Default depth image path
            default_depth_path = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "chillguy_depth.png")
            controlnet_image_paths = [default_depth_path]
            print(f"Using default ControlNet image: {controlnet_image_paths}")

    # Load ControlNet images from paths (just like input image)
    controlnet_images = None
    if use_controlnet and controlnet_image_paths:
        controlnet_images = []
        for image_path in controlnet_image_paths:
            try:
                control_image = Image.open(image_path).convert("RGB")
                controlnet_images.append(control_image)
                print(f"Loaded ControlNet image: {image_path}")
            except Exception as e:
                print(f"Error loading ControlNet image {image_path}: {e}")
                raise

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[22, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
        use_controlnet=use_controlnet,
        controlnet_model_ids=controlnet_model_ids,
        controlnet_scales=controlnet_scales,
        controlnet_images=controlnet_images,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    image_tensor = stream.preprocess_image(input)

    st = time.time()
    for _ in range(20):
        for _ in range(stream.batch_size - 1):
            stream(image=image_tensor)

        output_image = stream(image=image_tensor)
    print(f"Time taken: {stream.batch_size * 20 / (time.time() - st)} fps")
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
