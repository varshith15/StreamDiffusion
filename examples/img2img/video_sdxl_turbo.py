import os
import cv2
import sys
from typing import Literal, Dict, Optional

import fire
import time
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "xilin_move.mp4"),
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.mp4"),
    model_id_or_path: str = "stabilityai/sdxl-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "Naruto Uzumaki from the Naruto anime, spiky blond hair, orange ninja outfit with blue accents, blue headband with Konoha leaf symbol, cinematic anime style, energetic action pose, high detail, vibrant colors, dramatic lighting",
    negative_prompt: str = "black and white, blurry, low resolution, pixelated, pixel art, low quality, low fidelity",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    use_denoising_batch: bool = True,
    guidance_scale: float = 0.0,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 478,
    delta: float = 0.5,
):
    if guidance_scale <= 1.0:
        cfg_type = "none"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[25, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
        output_type="np",
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {frame_width}x{frame_height} @ {fps_input:.2f} FPS, {total_frames} frames")

    combined_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps_input, (combined_width, height))

    print("Warming up the model...")
    ret, frame = cap.read()
    if ret:
        frame_resized = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(frame_rgb)
        
        for _ in range(5):
            for _ in range(stream.batch_size - 1):
                image_tensor = stream.preprocess_image(input_image)
                stream(image=image_tensor)
            image_tensor = stream.preprocess_image(input_image)
            _ = stream(image=image_tensor)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    inference_time_sum = 0
    
    print("Starting video processing...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(frame_rgb)

        inference_start = time.time()
        image_tensor = stream.preprocess_image(input_image)
        output_image = stream(image=image_tensor)
        inference_time = time.time() - inference_start
        inference_time_sum += inference_time

        if output_image.dtype != 'uint8':
            output_image = (output_image * 255).astype('uint8')
        
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        input_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        combined_frame = np.hstack([input_bgr, output_bgr])
        out.write(combined_frame)

        frame_count += 1

        if frame_count % 30 == 0:
            avg_inference_fps = frame_count / inference_time_sum if inference_time_sum > 0 else 0
            elapsed_time = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            eta = (elapsed_time / frame_count) * (total_frames - frame_count)
            
            print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Inference FPS: {avg_inference_fps:.2f} | ETA: {eta:.1f}s")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_inference_fps = frame_count / inference_time_sum if inference_time_sum > 0 else 0

    print(f"\nVideo processing completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average overall FPS: {avg_fps:.2f}")
    print(f"Average inference FPS: {avg_inference_fps:.2f}")
    print(f"Output saved to: {output}")

    return {"total_time": total_time, "avg_fps": avg_fps, "avg_inference_fps": avg_inference_fps}


if __name__ == "__main__":
    metrics = {}
    for size in [512, 640, 704, 768, 896, 1024]:
        metrics[size] = main(width=size, height=size, output=os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", f"{size}x{size}.mp4"))

    print(metrics)
