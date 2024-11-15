import os
import torch
import onnx
from pathlib import Path
from utils import tools
from PIL import Image
# from ip_adapter import IPAdapterXL

# # Khởi tạo pipeline với các tham số phù hợp
# pipeline = tools.get_pipeline(
#     "neta-art/neta-xl-2.0",
#     "Eugeoter/controlnext-sdxl-anime-canny",
#     "Eugeoter/controlnext-sdxl-anime-canny",
#     vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
#     lora_path=None,
#     load_weight_increasement=False,
#     enable_xformers_memory_efficient_attention=False,
#     revision=None,
#     variant=None,
#     hf_cache_dir=None,
#     use_safetensors=True,
#     device='cuda',
# )

# pipeline.to("cuda")
# #pipeline.enable_xformers_memory_efficient_attention()
# pipeline.enable_vae_slicing()

# validation_image = Image.open("/home/tiennv/duypc/TensorRT/demo/Diffusion/ControlNeXt-SDXL/examples/anime_canny/condition_0.jpg").convert("RGB")

# from utilities import load_calib_prompts

# cali_prompts = load_calib_prompts(batch_size=2, calib_data_path="./abc.txt")
 
# # Create the int8 quantization recipe
# from utilities import get_smoothquant_config
# quant_config = get_smoothquant_config(pipeline.unet, quant_level=3.0)
 
# def do_calibrate(base, calibration_prompts, **kwargs):
#     for i_th, prompts in enumerate(calibration_prompts):
#         if i_th >= kwargs["calib_size"]:
#             return
#         base(
#             prompt=prompts,
#             controlnet_image=validation_image,
#             controlnet_scale=1.0,
#             num_inference_steps=kwargs["n_steps"],
            
#             negative_prompt=[
#                 "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
#             ]
#             * len(prompts),
#         ).images
                                
# def calibration_loop():
#     do_calibrate(
#         base=pipeline,
#         calibration_prompts=cali_prompts,
#         calib_size=384,
#         n_steps=10,
#     )
#     # Apply the quantization recipe and run calibration  
# import ammo.torch.quantization as atq 
# quantized_model = atq.quantize(pipeline.unet, quant_config, forward_loop = calibration_loop)
 
# # Save the quantized model
# import ammo.torch.opt as ato
# ato.save(quantized_model, 'base.unet.int8.pt')



from model.unet import UNet2DConditionModel

device = "cuda"

unet = UNet2DConditionModel.from_pretrained(
        "neta-art/neta-xl-2.0", 
        subfolder="unet", 
    ).to(device, dtype=torch.float16)

sample = torch.randn((1, 4, 128, 128), dtype=torch.float16, device=device)
timestep = torch.rand(1, dtype=torch.float16, device=device)
encoder_hidden_state = torch.randn((1, 81, 2048), dtype=torch.float16, device=device)
mid_block_additional_residual_scale = torch.tensor([1], dtype=torch.float16, device=device)
mid_block_additional_residual = torch.randn((1, 320, 128, 128), dtype=torch.float16, device=device)

from utilities import load_calib_prompts

cali_prompts = load_calib_prompts(batch_size=2, calib_data_path="./abc.txt")
 
# Create the int8 quantization recipe
from utilities import get_smoothquant_config
quant_config = get_smoothquant_config(unet, quant_level=3.0)
 
def do_calibrate(base, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        base(
            sample,
            timestep,
            encoder_hidden_state,
            mid_block_additional_residual,
            mid_block_additional_residual_scale
        )
                                
def calibration_loop():
    do_calibrate(
        base=unet,
        calibration_prompts=cali_prompts,
        calib_size=384,
        n_steps=10,
    )
    # Apply the quantization recipe and run calibration  
    
import ammo.torch.quantization as atq 
quantized_model = atq.quantize(unet, quant_config, forward_loop = calibration_loop)
 
# Save the quantized model
import ammo.torch.opt as ato
ato.save(quantized_model, 'base.unet.int8.pt')

from utilities import filter_func, quantize_lvl
import ammo.torch.opt as ato

torch.cuda.empty_cache()

# unet = ato.restore(unet, 'base.unet.int8.pt')
quantize_lvl(unet, quant_level=3.0)
atq.disable_quantizer(unet, filter_func) 

import onnx
from pathlib import Path


output_path = Path('/home/tiennv/duypc/TensorRT/demo/Diffusion/onnx_unet')
output_path.mkdir(parents=True, exist_ok=True)

dummy_inputs = (sample, timestep, encoder_hidden_state, mid_block_additional_residual, mid_block_additional_residual_scale)

onnx_output_path = output_path / "unet" / "model.onnx"
onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    unet,
    dummy_inputs,         
    str(onnx_output_path),  
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['sample', 'timestep', 'encoder_hidden_state', 'control_out', 'control_scale'],   
    output_names=['predict_noise'],  
    dynamic_axes={
        "sample": {0: "B"},
        "encoder_hidden_state": {0: "B", 1: "1B", 2: '2B'},  
        "control_out": {0: "B"},
        "predict_noise": {0: 'B'}
    }
)

# Tối ưu hóa và lưu mô hình ONNX
unet_opt_graph = onnx.load(str(onnx_output_path))
unet_optimize_path = output_path / "unet_optimize"
unet_optimize_path.mkdir(parents=True, exist_ok=True)
unet_optimize_file = unet_optimize_path / "model.onnx"

onnx.save_model(
    unet_opt_graph,
    str(unet_optimize_file),  
    save_as_external_data=True, 
    all_tensors_to_one_file=True,  
    location="weights.pb", 
)

