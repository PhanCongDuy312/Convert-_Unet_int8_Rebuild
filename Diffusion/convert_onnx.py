from model.unet import UNet2DConditionModel

device = "cuda"

unet = UNet2DConditionModel.from_pretrained(
        "neta-art/neta-xl-2.0", 
        subfolder="unet", 
    ).to(device)

from utilities import filter_func, quantize_lvl
import ammo.torch.opt as ato

unet = ato.restore(unet, 'base.unet.int8.pt')
quantize_lvl(unet, quant_level=3.0)
atq.disable_quantizer(unet, filter_func) 

import onnx
from pathlib import Path
output_path = Path('/home/tiennv/duypc/TensorRT/demo/Diffusion/onnx_unet')
output_path.mkdir(parents=True, exist_ok=True)

sample = torch.randn((1, 4, 128, 128), dtype=torch.float16, device=device)
timestep = torch.rand(1, dtype=torch.float16, device=device)
encoder_hidden_state = torch.randn((1, 81, 2048), dtype=torch.float16, device=device)
mid_block_additional_residual_scale = torch.tensor([1], dtype=torch.float16, device=device)
mid_block_additional_residual = torch.randn((1, 320, 128, 128), dtype=torch.float16, device=device)

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

