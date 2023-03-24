import os
from tqdm import tqdm
import numpy as np
import argparse

# import pandas as pd
import time as t
from time import time
import cv2
from loguru import logger
import dill
import numbers
from google.protobuf.json_format import MessageToDict
import json
from drawline import draw_rect
import math
from operator import itemgetter
from skimage.io import imsave
# import onnx

from common_cvinfer import *
YOLOV5_MODEL_LIST=['yolov5n','yolov5n6','yolov5s','yolov5s6','yolov5m','yolov5m6']
YOLOX_MODEL_LIST =['yolox_nano','yolox_tiny','yolox_s']

cachedEnable = 1
timeSleep = 10*60

# deviceSet according to execProvider
# deviceSet = "cpu"               # Default device for YOLOV5, other is gpu
# ["CPUExecutionProvider", "CUDAExecutionProvider"]
execProvider = "CPUExecutionProvider"

INTERPOLATIONS = {
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

Pose_num_kpt = 18

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])

def get_alpha(rate=30, cutoff=1):
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)

class LowPassFilter:
    def __init__(self):
        self.x_previous = None

    def __call__(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered

class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        self.x_previous = x
        return x_filtered

class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose_num_kpt)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose_num_kpt):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose_num_kpt, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)

class ImgClassOnnxModel:
    """
    base implementation of the onnx inference model for image classification
    """

    def __init__(self, onnx_path, execution_provider="CPUExecutionProvider"):
        # Load time recorder
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0

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

        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        startLoadEngine = time()
        self.session = ORT.InferenceSession(onnx_path, providers=[execution_provider,],)
        self.engineTime = time() - startLoadEngine

        self.input_name = self.session.get_inputs()[0].name

        # load config from json file
        # config_path is a json file
        config_file = onnx_path.replace(".onnx", ".configuration.json")
        assert os.path.exists(config_file)
        with open(config_file, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def getEngineTime(self):
        return self.engineTime

    @logger.catch
    def getpreProcessTime(self):
        return self.preProcessTime

    @logger.catch
    def getinferenceTime(self):
        return self.getinferenceTime

    @logger.catch
    def getpostProcessTime(self):
        return self.postProcessTime

    @logger.catch
    def preprocess(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)
        return self.preprocess_function(frame, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, model_output):
        return self.postprocess_function(model_output, self.config["postprocessing"])

    @logger.catch
    def __call__(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)

        # calling preprocess
        start = time()
        model_input, crop_size, meta_data = self.preprocess_function(
            frame, self.config["preprocessing"]
        )
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start

        # compute ONNX Runtime output prediction
        ort_inputs = {self.input_name: model_input}
        model_output = self.session.run(None, ort_inputs)
        inferenceTime = time()
        self.inferenceTime = inferenceTime - preProcessTime

        # calling postprocess
        categoryName, score = self.postprocess_function(
            model_output, self.config["postprocessing"]
        )
        postProcessTime = time()
        self.postProcessTime = postProcessTime - inferenceTime

        return categoryName, score

class ObjectDetectOnnxModel:
    def __init__(self, onnx_path, execution_provider="CPUExecutionProvider"):
        # Load time recorder
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0

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
        
        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        startLoadEngine = time()
        self.session = ORT.InferenceSession(onnx_path, providers=[execution_provider,],)
        self.engineTime = time() - startLoadEngine
        self.input_name = self.session.get_inputs()[0].name

        # load config from json file
        # config_path is a json file
        config_file = onnx_path.replace(".onnx", ".configuration.json")
        assert os.path.exists(config_file)
        with open(config_file, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def preprocess(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)
        return self.preprocess_function(frame, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, frame, model_output):
        return self.postprocess_function(
            frame, model_output, self.config["postprocessing"]
        )

    @logger.catch
    def __call__(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)

        # calling preprocess
        start = time()
        model_input = self.preprocess_function(frame, self.config["preprocessing"])
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start

        # compute ONNX Runtime output prediction
        ort_inputs = {self.input_name: model_input}
        model_output = self.session.run(None, ort_inputs)
        inferenceTime = time()
        self.inferenceTime = inferenceTime - preProcessTime

        # calling postprocess
        drawingFrameData = self.postprocess_function(
            model_input, model_output, self.config["postprocessing"]
        )
        postProcessTime = time()
        self.postProcessTime = postProcessTime - inferenceTime

        return drawingFrameData

class customObjectDetectOnnxModel:
    def __init__(self, onnx_path, execution_provider="CPUExecutionProvider"):
        # Load time recorder
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0
        # load preprocessing function
        preprocess_file = onnx_path.replace(".onnx", ".preprocess")
        # preprocess_file="/home/user/datTran/submission/testCVInfer/deploy/object_detect_custom_onnx/tinyYOLOv2.preprocess"
        assert os.path.exists(preprocess_file)
        with open(preprocess_file, "rb") as fid:
            self.preprocess_function = dill.load(fid)

        # similarly, load postprocessing
        # postprocess_file=preprocess_file.replace(".preprocess",".postprocess")
        postprocess_file = onnx_path.replace(".onnx", ".postprocess")
        assert os.path.exists(postprocess_file)
        with open(postprocess_file, "rb") as fid:
            self.postprocess_function = dill.load(fid)

        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        
        modelName = onnx_path.split('/')[-1].split('.')[0]
        self.modelName = modelName
        logger.info(f"ModelName: {modelName}")
        if modelName in YOLOV5_MODEL_LIST:

            # Change device set accourding to execution provider
            if execution_provider == "CPUExecutionProvider":
                deviceSet = "cpu"
            elif execution_provider == "CUDAExecutionProvider":
                deviceSet = "gpu"
            else:
                logger.warning("UNKNOWN EXEC PROVIDER")
                

            from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx
            self.session = Yolov5Onnx(
                classes="coco",
                backend="onnx",
                weight=onnx_path,
                device=deviceSet
            )
            self.isYOLOv5 = True
        else:
            startLoadEngine = time()
            onnx_path = preprocess_file.replace(".preprocess", ".onnx")
            self.session = ORT.InferenceSession(onnx_path, providers=[execution_provider,],)
            self.engineTime = time() - startLoadEngine
            self.input_name = self.session.get_inputs()[0].name
            self.isYOLOv5 = False


            # UNCOMMENT THIS BLOCK FOR DEBUGGING
            # print(onnx_path)
            # self.onnx_model = onnx.load(onnx_path)
            # onnx.checker.check_model(self.onnx_model)
            # print(self.onnx_model.graph.input)
            # for _input in self.onnx_model.graph.input:
            #     dim = _input.type.tensor_type.shape.dim
            #     print(dim)
            #     onnxInputShape = [MessageToDict(d).get("dimValue") for d in dim]
            #     print(onnxInputShape)
            # onnxInputShape = onnxInputShape[-2:]
            # logger.warning(onnxInputShape)

        # load config from json file
        # config_path is a json file
        config_file = onnx_path.replace(".onnx", ".configuration.json")
        assert os.path.exists(config_file)
        with open(config_file, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def preprocess(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)
        return self.preprocess_function(frame, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, frame, model_output):
        return self.postprocess_function(
            model_output, frame, self.config["postprocessing"]
        )

    @logger.catch
    def __call__(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)

        # calling preprocess
        start = time()
        model_input, _, self.frameToDraw = self.preprocess_function(
            frame, self.config["preprocessing"]
        )
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start

       # compute ONNX Runtime output prediction
        if self.isYOLOv5:
            # For YOLOv5 models
            ort_input = model_input
            model_output = self.session(ort_input)
            inferenceTime = time()
            self.inferenceTime = inferenceTime - preProcessTime

            drawingFrameData = self.postprocess_function(
                model_output,self.frameToDraw, self.config["postprocessing"]
            )
            postProcessTime = time()
            self.postProcessTime = postProcessTime - inferenceTime

        else:
            if self.modelName in ['tinyYOLOv3']:
                
                batchTmpSize = np.array([self.frameToDraw.shape()[1],self.frameToDraw.shape()[0]],dtype=np.float32).reshape(1,2)
                input2_name = self.session.get_inputs()[1].name
                model_output = self.session.run(None,{
                    self.input_name : model_input, 
                    input2_name: batchTmpSize
                })
                inferenceTime = time()
                self.inferenceTime = inferenceTime - preProcessTime

                # calling postprocess
                drawingFrameData = self.postprocess_function(
                    model_output, self.frameToDraw, self.config["postprocessing"]
                )

                # drawingFrameData = yolov3_ObjectDetectPostProcess(
                #     model_output, self.frameToDraw, self.config["postprocessing"]
                # )
                postProcessTime = time()
                self.postProcessTime = postProcessTime - inferenceTime
            
            else:
                ort_inputs = {self.input_name: model_input}
                # model_output here shape example (1,1,25,13,13)
                model_output = self.session.run(None, ort_inputs)
                inferenceTime = time()
                self.inferenceTime = inferenceTime - preProcessTime

                # calling postprocess
                drawingFrameData = self.postprocess_function(
                    model_output[0][0], self.frameToDraw, self.config["postprocessing"]
                )
                postProcessTime = time()
                self.postProcessTime = postProcessTime - inferenceTime

        return drawingFrameData

class yoloxObjectDetectOnnxModel:

    def __init__(self, onnx_path, execution_provider="CPUExecutionProvider"):
        # Load time recorder
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0
        # load preprocessing function
        preprocess_file = onnx_path.replace(".onnx", ".preprocess")
        # preprocess_file="/home/user/datTran/submission/testCVInfer/deploy/object_detect_custom_onnx/tinyYOLOv2.preprocess"
        assert os.path.exists(preprocess_file)
        with open(preprocess_file, "rb") as fid:
            self.preprocess_function = dill.load(fid)

        # similarly, load postprocessing
        # postprocess_file=preprocess_file.replace(".preprocess",".postprocess")
        postprocess_file = onnx_path.replace(".onnx", ".postprocess")
        logger.warning(postprocess_file)
        assert os.path.exists(postprocess_file)
        with open(postprocess_file, "rb") as fid:
            self.postprocess_function = dill.load(fid)

        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        
        modelName = onnx_path.split('/')[-1].split('.')[0]
        logger.info(f"ModelName: {modelName}")

        startLoadEngine = time()
        onnx_path = preprocess_file.replace(".preprocess", ".onnx")
        self.session = ORT.InferenceSession(onnx_path, providers=[execution_provider,],)
        self.engineTime = time() - startLoadEngine
        self.input_name = self.session.get_inputs()[0].name
  
        # load config from json file
        # config_path is a json file
        config_file = onnx_path.replace(".onnx", ".configuration.json")
        assert os.path.exists(config_file)
        with open(config_file, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def preprocess(self, image):
        return self.preprocess_function(image,config, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, model_output, image, meta_data):
        return self.postprocess_function(
            model_output, image, self.config["postprocessing"],meta_data
        )

    @logger.catch
    def __call__(self, cv2Img):

        # Create meta_data for temporty isage
        self.meta_data = {}
        height_after = self.config["preprocessing"]["new_height"]
        width_after = self.config["preprocessing"]["new_width"]

        self.meta_data['height_after'] = height_after
        self.meta_data['width_after'] = width_after

        # logger.info(self.meta_data)

        # calling preprocess
        start = time()
        model_input, self.meta_data['resize_ratio'] = self.preprocess_function(
            cv2Img, self.config["preprocessing"]
        )
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start
       # compute ONNX Runtime output prediction
        ort_inputs = {self.input_name: model_input[None, :, :, :]}
        # model_output here shape example (1,1,25,13,13)
        model_output = self.session.run(None, ort_inputs)
        inferenceTime = time()
        self.inferenceTime = inferenceTime - preProcessTime

        # calling postprocess
        drawingFrameData = self.postprocess_function(
            model_output, cv2Img, self.config["postprocessing"],self.meta_data
        )
        postProcessTime = time()
        self.postProcessTime = postProcessTime - inferenceTime

        return drawingFrameData

class HumanPoseOnnxModel:
    def __init__(self, onnx_path, execution_provider="CPUExecutionProvider"):
        # Load time recorder
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0

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

        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        startLoadEngine = time()
        self.session = ORT.InferenceSession(onnx_path, providers=[execution_provider,],)
        self.engineTime = time() - startLoadEngine

        self.input_name = self.session.get_inputs()[0].name

        # load config from json file
        # config_path is a json file
        config_file = onnx_path.replace(".onnx", ".configuration.json")
        assert os.path.exists(config_file)
        with open(config_file, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def getEngineTime(self):
        return self.engineTime

    @logger.catch
    def getpreProcessTime(self):
        return self.preProcessTime

    @logger.catch
    def getinferenceTime(self):
        return self.getinferenceTime

    @logger.catch
    def getpostProcessTime(self):
        return self.postProcessTime

    @logger.catch
    def preprocess(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)
        return self.preprocess_function(frame, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, model_output):
        return self.postprocess_function(model_output, self.config["postprocessing"])

    @logger.catch
    def __call__(self, frame: Frame, cv2Img):
        # input must be a frame
        assert isinstance(frame, Frame)

        # calling preprocess
        start = time()
        model_input,  self.meta_data = self.preprocess_function(
            frame, self.config["preprocessing"]
        )
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start

        # compute ONNX Runtime output prediction
        ort_inputs = {self.input_name: model_input}
        model_output = self.session.run(None, ort_inputs)
        inferenceTime = time()
        self.inferenceTime = inferenceTime - preProcessTime
        
        # calling postprocess
        # dump counter 
        counter=0
        dumpOut = self.postprocess_function(
            model_output, cv2Img, self.meta_data, self.config["postprocessing"],counter
        )
        postProcessTime = time()
        self.postProcessTime = postProcessTime - inferenceTime

class SemanSegmenOnnxModel:
    
    def __init__(self, onnx_path, execution_provider="CPUExecutionProvider"):
        # Load time recorder
        self.preProcessTime = 0.0
        self.inferenceTime = 0.0
        self.postProcessTime = 0.0
        self.engineTime = 0.0

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

        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        startLoadEngine = time()
        self.session = ORT.InferenceSession(onnx_path, providers=[execution_provider,],)
        self.engineTime = time() - startLoadEngine

        self.input_name = self.session.get_inputs()[0].name

        # load config from json file
        # config_path is a json file
        config_file = onnx_path.replace(".onnx", ".configuration.json")
        assert os.path.exists(config_file)
        with open(config_file, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def getEngineTime(self):
        return self.engineTime

    @logger.catch
    def getpreProcessTime(self):
        return self.preProcessTime

    @logger.catch
    def getinferenceTime(self):
        return self.getinferenceTime

    @logger.catch
    def getpostProcessTime(self):
        return self.postProcessTime

    @logger.catch
    def preprocess(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)
        return self.preprocess_function(frame, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, model_output):
        return self.postprocess_function(model_output, self.config["postprocessing"])

    @logger.catch
    def __call__(self, frame: Frame):
        # input must be a frame
        assert isinstance(frame, Frame)

        # calling preprocess
        start = time()
        model_input, meta_data = self.preprocess_function(
            frame, self.config["preprocessing"]
        )
        preProcessTime = time()
        self.preProcessTime = preProcessTime - start

        # compute ONNX Runtime output prediction
        ort_inputs = {self.input_name: model_input}
        model_output = self.session.run(None, ort_inputs)
        inferenceTime = time()
        self.inferenceTime = inferenceTime - preProcessTime

        # calling postprocess
        dumpOut = self.postprocess_function(
            model_output, self.config["postprocessing"]
        )
        postProcessTime = time()
        self.postProcessTime = postProcessTime - inferenceTime

try:
    import onnxruntime as ORT
except ImportError as e:
    raise ImportError(f"Please install onnx and onnxruntime first. {e}")


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
    output_dir = f"{cwd}/csv_output"

    modelOnnxPathName = os.path.join(modelDir, modelFn + ".onnx")
    # logger.info(modelOnnxPathName)
    if not os.path.exists(COCO_verify_dir):
        msg = (
            'Cannot find "COCO_5000_imgs" directory.',
            "Please download COCO image dataset first",
        )
        raise RuntimeError(msg)

    if cachedEnable:
        csvOutDir = f"{cwd}/cache"
    else:
        csvOutDir = f"{cwd}/no_cache"
    
    if not os.path.exists(csvOutDir):
        msg = (
            f'Cannot find {csvOutDir}" directory.',
            "Please check extractDrive.py again",
        )
        raise RuntimeError(msg)

    if execProvider== "CPUExecutionProvider":
        csvOutDir = os.path.join(csvOutDir,"cpu")
    elif execProvider== "CUDAExecutionProvider":
        csvOutDir = os.path.join(csvOutDir,"gpu")
    else:
        msg = (f'Unknown execution provider. Check it again')
        raise RuntimeError(msg)
    
    if not os.path.exists(csvOutDir):
        os.mkdir(csvOutDir)


    isImgClassApplication = False
    isObjDetectApplication = False
    isYOLOXRunning = False
    isHumanPoseApplication = False
    isSemanSegmen = False

    if application == "image_class":
        onnx_model = ImgClassOnnxModel(onnx_path=modelOnnxPathName,execution_provider=execProvider)
        isImgClassApplication = True

    elif application == "object_detect":
        onnx_model = ObjectDetectOnnxModel(onnx_path=modelOnnxPathName,execution_provider=execProvider)
        isObjDetectApplication = True

    elif application == "object_detect_custom":
        logger.warning(modelOnnxPathName)
        onnx_model = customObjectDetectOnnxModel(onnx_path=modelOnnxPathName,execution_provider=execProvider)
        isObjDetectApplication = True

    elif application == "object_detect_yolox":
        onnx_model = yoloxObjectDetectOnnxModel(onnx_path=modelOnnxPathName,execution_provider=execProvider)
        isYOLOXRunning = True

    elif application == "human_pose":
        logger.warning(modelOnnxPathName)
        onnx_model = HumanPoseOnnxModel(onnx_path=modelOnnxPathName,execution_provider=execProvider)
        isHumanPoseApplication = True

    elif application == "seman_segmen":
        onnx_model = SemanSegmenOnnxModel(onnx_path=modelOnnxPathName,execution_provider=execProvider)
        isSemanSegmenApplication = True

    if isObjDetectApplication:
        drawingResultDir = f"{cwd}/drawingRes/{modelFn}"
        if not os.path.exists(drawingResultDir):
            logger.warning(
                f"Creating drawing object detection directory {drawingResultDir}..."
            )
            os.mkdir(drawingResultDir)
    
    # WARM UP MODEL FOR CACHE LOAD
    if cachedEnable:
        for i in range(100):
            
            imgIdx = os.path.join(COCO_verify_dir, f"{i}.jpg")
            cv2Img = cv2.imread(imgIdx)
            tempImg = Frame(imgIdx)

            assert isImgClassApplication is True \
            or isObjDetectApplication is True \
            or isYOLOXRunning is True \
            or isHumanPoseApplication is True \
            or isSemanSegmenApplication is True

            if isImgClassApplication:
                classOut, scoreOut = onnx_model(tempImg)
            elif isObjDetectApplication:
                drawingFrameOut = onnx_model(tempImg)
            elif isYOLOXRunning:
                drawingFrameOut = onnx_model(cv2Img)
            elif isHumanPoseApplication:
                dumpOut = onnx_model(tempImg,cv2Img)
            elif isSemanSegmenApplication:
                dumpOut = onnx_model(tempImg)

    scoreClasstDict = {}
    timeBenchmarkList = []
    engineLoadTime = onnx_model.engineTime
    for i in tqdm(range(numIteration)):
        imgIdx = os.path.join(COCO_verify_dir, f"{i}.jpg")
        tempImg = Frame(imgIdx) 
        cv2Img = cv2.imread(imgIdx)

        assert isImgClassApplication is True \
        or isObjDetectApplication is True \
        or isYOLOXRunning is True \
        or isHumanPoseApplication is True \
        or isSemanSegmenApplication is True 

        if isImgClassApplication:
            classOut, scoreOut = onnx_model(tempImg)
            scoreClasstDict[classOut] = scoreOut
        elif isObjDetectApplication:
            drawingFrameOut = onnx_model(tempImg)

            # Drawing image to "drawingRes" directory
            assert os.path.exists(drawingResultDir)
            imgPath = os.path.join(drawingResultDir, f"{i}.jpg")
            # logger.warning(imgPath)
            # imsave(imgPath, drawingFrameOut)
        elif isYOLOXRunning:
            # For YOLOX models
            drawingFrameOut = onnx_model(cv2Img)
        elif isHumanPoseApplication:
            dumpOut = onnx_model(tempImg,cv2Img)
        elif isSemanSegmenApplication:
            dumpOut = onnx_model(tempImg)
        else:
            logger.warning("At least one application must be specified")

        preProcessTime, inferenceTime, postProcessTime = (
            onnx_model.preProcessTime,
            onnx_model.inferenceTime,
            onnx_model.postProcessTime,
        )

        tempTimeList = [preProcessTime, inferenceTime, postProcessTime]
        timeBenchmarkList.append(tempTimeList)

    # logger.info("Finish benchmarking!!")
    timeBenchmarkArr = np.array(timeBenchmarkList)
    timeBenchamrkSum = timeBenchmarkArr.sum(axis=0)
    preProTime, inferTime, postProTime = (
        timeBenchamrkSum[0],
        timeBenchamrkSum[1],
        timeBenchamrkSum[2],
    )
    logger.info(
        f"Pre: {preProTime}, Inf: {inferTime}, Post: {postProTime}, Engine: {engineLoadTime}"
    )
    
    csvFileOut = modelFn + "_onnx" + ".csv"
    csvOutDir = os.path.join(csvOutDir, csvFileOut)
    # timeBenchmarkDF = pd.DataFrame(timeBenchmarkArr)
    # timeBenchmarkDF.to_csv(output_dir, header=False, index=False)
    np.savetxt(csvOutDir, timeBenchmarkArr, delimiter=",")
    print(f"Finish exporting result to file {csvFileOut}.")

    if cachedEnable:
        logger.info("Cache enabled. Sleeping for 10 mins...")
        t.sleep(timeSleep)
