import re
import torch
from PIL import Image
from typing import Any
import torch.nn.functional as F
import modelopt.torch.quantization as mtq
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from ...pipeline import StreamDiffusion

from .plugin_calib import PercentileCalibrator

FP8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

def fp8_mha_disable(backbone, quantized_mha_output: bool = True):
    def mha_filter_func(name):
        pattern = re.compile(
            r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer).*"
            if quantized_mha_output
            else r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
        )
        return pattern.match(name) is not None

    if hasattr(F, "scaled_dot_product_attention"):
        mtq.disable_quantizer(backbone, mha_filter_func)

def set_quant_config_attr(quant_config, trt_high_precision_dtype, quant_algo, **kwargs):
    algo_cfg = {"method": quant_algo}

    if quant_algo == "smoothquant" and "alpha" in kwargs:
        algo_cfg["alpha"] = kwargs["alpha"]
    elif quant_algo == "svdquant" and "lowrank" in kwargs:
        algo_cfg["lowrank"] = kwargs["lowrank"]
    quant_config["algorithm"] = algo_cfg

    for p in quant_config["quant_cfg"].values():
        if "num_bits" in p and "trt_high_precision_dtype" not in p:
            p["trt_high_precision_dtype"] = trt_high_precision_dtype

def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|proj_out).*"
    )
    return pattern.match(name) is not None

def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="min-mean",
):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "*output_quantizer": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config

def check_lora(unet):
    for name, module in unet.named_modules():
        if isinstance(module, (LoRACompatibleConv, LoRACompatibleLinear)):
            assert module.lora_layer is None, (
                f"To quantize {name}, LoRA layer should be fused/merged. Please"
                " fuse the LoRA layer before quantization."
            )

def quantize_lvl(backbone, quant_level=2.5, linear_only=False, enable_conv_3d=True):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if linear_only:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            else:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ) and name != "proj_out":  # Disable the final output layer from flux model
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, torch.nn.Conv3d) and not enable_conv_3d:
            """
                Error: Torch bug, ONNX export failed due to unknown kernel shape in QuantConv3d.
                TRT_FP8QuantizeLinear and TRT_FP8DequantizeLinear operations in UNetSpatioTemporalConditionModel for svd
                cause issues. Inputs on different devices (CUDA vs CPU) may contribute to the problem.
            """
            module.input_quantizer.disable()
            module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            # TRT only supports FP8 MHA with head_size % 16 == 0.
            head_size = int(module.inner_dim / module.heads)
            if quant_level >= 4 and head_size % 16 == 0:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
                module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", False)
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
                module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", True)

def generate_fp8_scales(backbone):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax") and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (127 / 448.0)

def do_calibrate(
    pipe: StreamDiffusion,
    calibration_prompts: list[str],
    calibration_images: list[Image.Image],
) -> None:
    """
    Run calibration steps on the pipeline using the given prompts.
    """
    for _ in range(32):
        pipe.prepare(
            "elon musk",
            "",
            num_inference_steps=50,
            guidance_scale=1.2,
            delta=1.0,
        )
        
        # Run multiple calibration steps to collect enough data points
        total_steps = pipe.denoising_steps_num
        for _ in range(total_steps):
            for prompt, image in zip(calibration_prompts, calibration_images):
                pipe.update_prompt(prompt)
                image = pipe.image_processor.preprocess(image, 512, 512).to(device=pipe.device, dtype=pipe.dtype)
                pipe(image)


def quantize(pipe, int8: bool, fp8: bool):
    quant_level = 3.0
    calibration_prompts = ["elon musk"]
    calibration_images = [Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))]
    collect_method = "global_min"
    quant_algo = "smoothquant"
    percentile = 1.0
    alpha = 0.8
    n_steps = pipe.denoising_steps_num
    lowrank = 32
    trt_high_precision_dtype="Half"

    if int8:
        quantization_format = "int8"
    elif fp8:
        quantization_format = "fp8"

    if quantization_format == "int8":
        if collect_method == "default":
            raise ValueError(
                    "You must specify an explicit --collect-method (e.g., 'global_min') for int8."
                )
        if quant_algo != "smoothquant":
            raise ValueError(
                "INT8 quantization only works well when combined with SmoothQuant;"
                "otherwise, it will produce very poor quality results."
            )
        quant_config = get_int8_config(
            pipe.unet,
            quant_level,
            alpha,
            percentile,
            n_steps,
            collect_method=collect_method,
        )
    elif quantization_format == "fp8":
        if collect_method != "default":
            raise NotImplementedError("Only 'default' collect method is implemented for fp8.")
        quant_config = FP8_DEFAULT_CONFIG
    else:
        raise ValueError(f"Unsupported quantization format: {quantization_format}")

    set_quant_config_attr(
        quant_config,   
        trt_high_precision_dtype,
        quant_algo,
        alpha=alpha,
        lowrank=lowrank,
    )

    def forward_loop(mod):
        pipe.unet = mod
        do_calibrate(
            pipe=pipe,
            calibration_prompts=calibration_prompts,
            calibration_images=calibration_images,
        )
    
    backbone = pipe.unet
    check_lora(backbone)
    mtq.quantize(backbone, quant_config, forward_loop)
    quantize_lvl(backbone, quant_level)
    mtq.disable_quantizer(backbone, filter_func)

    if quantization_format == "fp8":
        generate_fp8_scales(backbone)

    torch.cuda.empty_cache()
    backbone.to("cuda")
    if quant_level == 4.0:
        fp8_mha_disable(backbone, quantized_mha_output=False)

    return backbone
    