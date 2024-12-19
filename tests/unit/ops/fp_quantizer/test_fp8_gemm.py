# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed import get_accelerator
from deepspeed.linear import QuantizationConfig

# TODO: [SW-208941] clear gaudi specific code.
from tests.unit.util import get_hpu_dev_version

# [SW-209231] Enable gp8_gemm test
pytest.skip("fp8 gemm (fp8 weight, float 16 input) is currently unimplemented", allow_module_level=True)

from deepspeed.ops.fp_quantizer import FP_Quantize, matmul_fp8


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_bits", [8], ids=[
    "qbits8",
])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048])
def test_fp_quant(dtype, q_bits, M):
    device_name = get_accelerator().device_name()
    quantization_group_size = 128

    quant_config = QuantizationConfig()
    quant_config.q_range_dtype = torch.float8_e4m3fn
    quant_config.q_dtype = torch.float8_e4m3fn
    # TODO: [SW-208941] clear gaudi specific code.
    if get_hpu_dev_version().lower() == 'gaudi2':
        quant_config.q_range_dtype = torch.float8_e4m3fnuz
    quant_config.group_size = quantization_group_size
    fpq = FP_Quantize(quantization_config=quant_config)

    N = 8192
    H = 4096

    x = torch.randn(M, H, dtype=dtype, device=device_name)
    weight_bf16 = torch.randn(H, N, dtype=dtype, device=device_name)

    weight, _ = fpq.quantize(weight_bf16.data, q_bits=q_bits, return_meta_tensor=True)
    scale = fpq.get_scales()
    out = matmul_fp8(
        x,
        weight,
        scale,
        quantization_group_size,
    )

    out_q = torch.matmul(x, fpq.dequantize(weight, scale=fpq.scale))

    error = ((out - out_q).abs() / (out.abs() + 1e-5)).sum() / out.numel()
    assert 0.004 > error, f"failed on batch-size {M} with error {error}"
