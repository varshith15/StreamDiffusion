import io
import os
import sys
from typing import List, Literal, Optional, Dict

import fire
import PIL.Image
import requests
import torch
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(
    iterations: int = 100,
    model_id_or_path: str = "stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "bad image , bad quality",
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    width: int = 512,
    height: int = 512,
    warmup: int = 10,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    device_ids: Optional[List[int]] = None,
    use_denoising_batch: bool = True,
    seed: int = 2,
):
    """
    Initializes the StreamDiffusionWrapper.

    Parameters
    ----------
    iterations : int, optional
        The number of iterations to run, by default 100.
    model_id_or_path : str
        The model id or path to load.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str, optional
        The prompt to use, by default "1girl with brown dog hair, thick glasses, smiling".
    negative_prompt : str, optional
        The negative prompt to use, by default "bad image , bad quality".
    use_lcm_lora : bool, optional
        Whether to use LCM-LoRA or not, by default True.
    use_tiny_vae : bool, optional
        Whether to use TinyVAE or not, by default True.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    warmup : int, optional
        The number of warmup steps to perform, by default 10.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    device_ids : Optional[List[int]], optional
        The device ids to use for DataParallel, by default None.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """       
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[32, 45],
        lora_dict=lora_dict,
        mode="img2img",
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=warmup,
        acceleration=acceleration,
        device_ids=device_ids,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        enable_similar_image_filter=False,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        cfg_type="initialize",  # initialize, full, self , none
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.4,
        delta=0.5,
    )

    downloaded_image = download_image("https://github.com/ddpn08.png").resize(
        (width, height)
    )

    # warmup
    for _ in range(warmup):
        image_tensor = stream.preprocess_image(downloaded_image)
        stream(image=image_tensor)

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in tqdm(range(iterations)):
        start.record()
        image_tensor = stream.preprocess_image(downloaded_image)
        stream(image=image_tensor)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))
    
    

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")
    import numpy as np

    fps_arr = 1000 / np.array(results)
    print(f"Max FPS: {np.max(fps_arr)}")
    print(f"Min FPS: {np.min(fps_arr)}")
    print(f"Std: {np.std(fps_arr)}")


if __name__ == "__main__":
    fire.Fire(run)
