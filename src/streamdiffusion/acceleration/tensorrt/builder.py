import gc
import os
from typing import *
from pathlib import Path

import torch
from onnx import shape_inference

from .models import BaseModel
from .utilities import (
    build_engine,
    export_onnx,
    optimize_onnx,
)
from ...pipeline import StreamDiffusion


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        engine_path: str,
        opt_image_height: int = 512,
        opt_image_width: int = 512,
        opt_batch_size: int = 1,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = False,
        build_all_tactics: bool = False,
        onnx_opset: int = 20,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
        pipe: StreamDiffusion = None,
        int8: bool = None,
        fp8: bool = None,
        strongly_typed: bool = None,
        precision_constraints: str = None,
    ):
        if int8 or fp8:
            from .quantization import quantize
            pipe.unet = self.network
            self.network = quantize(pipe, int8, fp8)
            
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
        else:
            print(f"Exporting model: {onnx_path}")
            export_onnx(
                self.network,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            del self.network
            gc.collect()
            torch.cuda.empty_cache()
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"Found cached model: {onnx_opt_path}")
        else:
            print(f"Generating optimizing model: {onnx_opt_path}")
            onnx_model_path = str(Path(onnx_path).absolute().as_posix())
            shape_inference.infer_shapes_path(onnx_model_path, onnx_model_path)
            optimize_onnx(
                onnx_path=onnx_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
            )
        self.model.min_latent_shape = min_image_resolution // 8
        self.model.max_latent_shape = max_image_resolution // 8
        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
        else:
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
                # Pass quantization parameters
                int8=int8,
                fp8=fp8,
                strongly_typed=strongly_typed,
                precision_constraints=precision_constraints,
            )

        gc.collect()
        torch.cuda.empty_cache()
