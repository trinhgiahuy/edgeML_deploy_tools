import torch
import torchvision
import cv2
import numpy as np
import os
import tarfile
import dill
import matplotlib.pyplot as plt
from time import time
import numbers
import gdown
import argparse
from torchvision.io.image import read_image
from loguru import logger
from torchvision import models
from torchvision.models import *
from torchvision.models.detection import *
from tqdm import tqdm

from torchvision.io.image import read_image
from google.protobuf.json_format import MessageToDict
from constant import *
from common_cvinfer import *

import torchvision.transforms as T
import json

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

execProvider = "CPUExecutionProvider"


def buildTensorRTEngine(modelOnnxPathName):
    
    if not os.path.exists(modelOnnxPathName):
        print(
                "ONNX file {} not found, please generate it.".format(modelOnnxPathName)
            )
        exit(0)
    model_name = modelOnnxPathName.split('/')[-1]
    logger.warning(model_name)

    engineFilePath = modelOnnxPathName.replace('.onnx','.trt')
    if os.path.exists(engineFilePath):
        logger.info(f"Engine file for model {model_name} exists!")
        return

    
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        builder.max_batch_size = 1
        config.max_workspace_size = 1 << 28  # 256MiB


        print("Loading ONNX file from path {}...".format(modelOnnxPathName))
        with open(modelOnnxPathName, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
    
    
        logger.info(f'engineFilePath: {engineFilePath}')

        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(modelOnnxPathName))
        plan = builder.build_serialized_network(network, config)
        # engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engineFilePath, "wb") as f:
            f.write(plan)

def downloadEngine(modelName:str,application:str):
    engineURL = jetsonTensorLink[modelName]
    enginePath = f"{os.getcwd()}/{application}"
    if os.path.exists(enginePath):
        logger.warning("Application dir exist")
    else:
        os.mkdir(enginePath)

    engineFile = os.path.join(enginePath,modelName+".trt")
    logger.info(f"engineFile {engineFile}")
    if os.path.exists(engineFile):
        logger.warning(f"Engine file local exists")
        return engineFile

    # Start downloading engine 
    gdown.download(engineURL,engineFile,quiet=False)
    logger.info(f"Finish downloading engine file model {modelName}")

    return engineFile

def getEngine(modelName:str, application:str):

    # The downloadEngine will check if engine exists,
    # TODO: If not exist, build engine ??
    engineFile = downloadEngine(modelName=modelName,application=application)

    # if not os.path.exists(engineFilePath):
    #     print(
    #             "Engine file {} not found, please generate it.".format(engineFilePath)
    #         )
    #     exit(0)
        
    logger.info("Reading engine from file {}".format(engineFile))
    with open(engineFile, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    

class ImgClassTensorRTModel:
    """
    base implementation of tensorrt inference model
    """

    def __init__(self,onnx_path, tensor_engine,num_class: int = 100):
        
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0

        # TensorRT variable
        self.stream = None
        self.num_class = num_class
        self.engine = None
        self.context = None

        # load preprocessing function
        preprocess_file = onnx_path.replace(".onnx", ".preprocess")
        assert os.path.exists(preprocess_file)
        with open(preprocess_file, "rb") as fid:
            self.preprocess_function = dill.load(fid)

        # similarly, load postprocessing function
        postprocess_file = onnx_path.replace(".onnx", ".postprocess")
        assert os.path.exists(postprocess_file)
        with open(postprocess_file, "rb") as fid:
            self.postprocess_function = dill.load(fid)

        engineFilePath = onnx_path.replace('.onnx','.trt')
        self.engine = tensor_engine
        self.context = self.engine.create_execution_context()

    def allocate_memory(self, batch):

        self.output = np.empty(self.num_class, dtype=self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()        

    def __call__(self,frame: Frame):

        assert isinstance(frame, Frame)
        start = time()
        model_input, crop_size, meta_data = self.preprocess_function(
            frame, self.config["preprocessing"]
        )
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start

        if self.stream is None:
            self.allocate_memory(batch=model_input)
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, model_input, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        if hasattr(self.output,"eval"):
            logger.info("Calling model eval")
            self.output.eval()
        
        logger.info(type(self.output))
        logger.info(np.shape(self.output))

        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numIteration", type=int, default=5000, help="Number of iterations to run"
    )
    parser.add_argument(
        "--modelFn", type=str, required=True, help="File name of the model archive"
    )
    parser.add_argument(
        "--application",
        type=str,
        required=True,
        help="ML application (image_class,object_detect)",
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Runtime prefix (onnx,openVINO,tvm)"
    )
    args = parser.parse_args()
    numIteration = args.numIteration
    modelFn = args.modelFn
    application = args.application
    prefix = args.prefix

    cwd = os.getcwd()

    # Same directory with google drive
    predir = application
    COCO_verify_dir = f"{cwd}/COCO_5000_imgs"
    modelDir = f"{cwd}/{predir}_{prefix}"
    # output_dir = f"{cwd}/csv_output"
    tesor_output_dir = f"{cwd}/tensorRT/{execProvider}"

    modelOnnxPathName = os.path.join(modelDir, modelFn + ".onnx")
    modelTesorRTPathName = modelOnnxPathName
    # logger.info(modelOnnxPathName)
    if not os.path.exists(COCO_verify_dir):
        msg = (
            'Cannot find "COCO_5000_imgs" directory.',
            "Please download COCO image dataset first",
        )
        raise RuntimeError(msg)

    if not os.path.exists(tesor_output_dir):
        msg = (
            f'Cannot find {tesor_output_dir}" directory.',
            "Please check extractDrive.py again",
        )
        raise RuntimeError(msg)
    
    tensor_engine = getEngine(modelName=modelFn, application=application)
    # getTensorRTEngine(modelOnnxPathName=modelOnnxPathName)
    
    isImgClassApplication = False

    if application == "image_class":
        tensorRTModel = ImgClassTensorRTModel(onnx_path=modelOnnxPathName,tensor_engine=tensor_engine)
        isImgClassApplication = True


    timeBenchmarkList = []
    for i in tqdm(range(numIteration)):

        imgIdx = os.path.join(COCO_verify_dir, f"{i}.jpg")
        tempImg = Frame(imgIdx) 
        cv2Img = cv2.imread(imgIdx)

        if isImgClassApplication:
            dump = tensorRTModel(tempImg)

        preProcessTime, inferenceTime, postProcessTime = (
            tensorRTModel.preProcessTime,
            tensorRTModel.inferenceTime,
            tensorRTModel.postProcessTime,
        )
       
        tempTimeList = [preProcessTime, inferenceTime, postProcessTime]
        timeBenchmarkList.append(tempTimeList)

    timeBenchmarkArr = np.array(timeBenchmarkList)
    timeBenchamrkSum = timeBenchmarkArr.sum(axis=0)
    preProTime, inferTime, postProTime = (
        timeBenchamrkSum[0],
        timeBenchamrkSum[1],
        timeBenchamrkSum[2],
    )
    logger.info(
        f"Pre: {preProTime}, Inf: {inferTime}, Post: {postProTime}"
    )