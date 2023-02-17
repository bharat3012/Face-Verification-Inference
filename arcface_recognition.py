import onnx
import argparse
import math
import os
from typing import List, Tuple, Union
import logging
import tensorrt as trt
import sys

def remove_initializer_from_input(input, output):

    model = onnx.load(input)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, output)
    

def reshape(model, n: int = 1, h: int = 480, w: int = 640, mode='auto'):
    '''
    :param model: Input ONNX model object
    :param n: Batch size dimension
    :param h: Height dimension
    :param w: Width dimension
    :param mode: Set `retinaface` to reshape RetinaFace model, otherwise reshape Centerface
    :return: ONNX model with reshaped input and outputs
    '''
    if mode == 'auto':
        # Assert that retinaface models have outputs containing word 'stride' in their names

        out_name = model.graph.output[0].name
        if 'stride' in out_name.lower():
            mode = 'retinaface'
        elif out_name.lower() == 'fc1':
            mode = 'arcface'
        else:
            mode = 'centerface'

        input_name = model.graph.input[0].name
        out_shape = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        
        dyn_size = False
        if model.graph.input[0].type.tensor_type.shape.dim[2].dim_param == '?':
            dyn_size = True

        if input_name == 'input.1' and dyn_size is True:
            mode = 'scrfd'
        elif input_name == 'input.1' and out_shape == 512:
            mode = 'arcface'
        if  model.graph.input[0].type.tensor_type.shape.dim[3].dim_value == 3:
            mode = 'mask_detector'
        if len(model.graph.output) == 1 and len(model.graph.output[0].type.tensor_type.shape.dim) == 3:
            mode = 'yolov5-face'


    d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = n
    logging.debug(f"In shape: {d}")
    if mode != 'arcface':
        d[2].dim_value = h
        d[3].dim_value = w
    divisor = 4
    logging.debug(f"Mode: {mode}")
    if mode == 'yolov5-face':
        d = model.graph.output[0].type.tensor_type.shape.dim
        mx = (h * w) / 16
        s = mx - mx / 64
        d[0].dim_value = n
        d[1].dim_value = int(s)
        d[2].dim_value = 16
    elif mode != 'scrfd':
        for output in model.graph.output:
            if mode == 'retinaface':
                divisor = int(output.name.split('stride')[-1])
            d = output.type.tensor_type.shape.dim
            d[0].dim_value = n
            if mode not in  ('arcface', 'mask_detector'):
                d[2].dim_value = math.ceil(h / divisor)
                d[3].dim_value = math.ceil(w / divisor)
    logging.debug(f"Out shape: {d}")
    return model



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _build_engine_onnx(input_onnx: Union[str, bytes], force_fp16: bool = False, max_batch_size: int = 1,
                       max_workspace: int = 1024):

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        has_fp16 = builder.platform_has_fast_fp16
        if force_fp16 or has_fp16:
            logging.info('Building TensorRT engine with FP16 support.')
            if not has_fp16:
                logging.warning('Builder reports no fast FP16 support. Performance drop expected.')
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logging.warning('Building engine in FP32 mode.')

        config.max_workspace_size = max_workspace * 1024 * 1024

        if not parser.parse(input_onnx):
            print('ERROR: Failed to parse the ONNX')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

        if max_batch_size != 1:
            logging.warning('Batch size !=1 is used. Ensure your inference code supports it.')
        profile = builder.create_optimization_profile()
        # Get input name and shape for building optimization profile
        input = network.get_input(0)
        inp_shape = list(input.shape)
        inp_shape[0] = 1
        min_opt_shape = tuple(inp_shape)
        inp_shape[0] = max_batch_size
        max_shape = tuple(inp_shape)
        input_name = input.name
        profile.set_shape(input_name, min_opt_shape, min_opt_shape, max_shape)
        config.add_optimization_profile(profile)

        return builder.build_engine(network, config=config)


def check_fp16():
    builder = trt.Builder(TRT_LOGGER)
    has_fp16 = builder.platform_has_fast_fp16
    return has_fp16


def convert_onnx(input_onnx: Union[str, bytes], engine_file_path: str, force_fp16: bool = False,
                 max_batch_size: int = 1):
    '''
    Creates TensorRT engine and serializes it to disk
    :param input_onnx: Path to ONNX file on disk or serialized ONNX model.
    :param engine_file_path: Path where TensorRT engine should be saved.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :return: None
    '''

    onnx_obj = None
    if isinstance(input_onnx, str):
        with open(input_onnx, "rb") as f:
            onnx_obj = f.read()
    elif isinstance(input_onnx, bytes):
        onnx_obj = input_onnx

    engine = _build_engine_onnx(input_onnx=onnx_obj,
                                force_fp16=force_fp16, max_batch_size=max_batch_size)

    assert not isinstance(engine, type(None))

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
        
        
remove_initializer_from_input('arcface_r100_v1.onnx', 'arcface_r100_v1_clean.onnx')

im_size = [112, 112]
max_batch_size = 128
if max_batch_size !=1:
    batch_size=-1

model = onnx.load('arcface_r100_v1_clean.onnx')
reshaped = reshape(model, n=batch_size, h=im_size[1], w=im_size[0], mode='auto')

with open('arcface_r100_v1_dynamic.onnx', "wb") as file_handle:
    serialized = reshaped.SerializeToString()
    file_handle.write(serialized)
    
convert_onnx(serialized, 
             engine_file_path='arcface_r100_v1_dynamic.plan',
             max_batch_size=max_batch_size,
             force_fp16=True)