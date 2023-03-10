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

from skimage.io import imsave
# from common_cvinfer import *
YOLOV5_MODEL_LIST=['yolov5n','yolov5n6','yolov5s','yolov5s6','yolov5m','yolov5m6']
timeSleep = 5*60
deviceSet = "cpu"               # Default device for YOLOV5, other is gpu

INTERPOLATIONS = {
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

class Point:
    """
    abstraction for a point in an image
    a point must have non-negative coordinates
    """

    def __init__(self, x: int, y: int):
        # note x, y correspond to points on the width and height axis
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert x >= 0
        assert y >= 0
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def translate(self, reference):
        """
        change the coordinate system so that the current origin (0, 0) has a coordinate equal to
        `reference` point in the new system

        this is useful when mapping from a coorindate in a cropped image to the coorindate in
        the original image --> we need to pass the top left point as the reference point
        """
        return Point(self.x + reference.x, self.y + reference.y)

    def __str__(self):
        return "Point(x={}, y={})".format(self.x(), self.y())

class Color:
    """
    abstraction for color representation
    """

    def __init__(self, red=None, green=None, blue=None):
        if red is None:
            red = np.random.randint(0, 256)

        if green is None:
            green = np.random.randint(0, 256)

        if blue is None:
            blue = np.random.randint(0, 256)

        assert isinstance(red, int) and 0 <= red <= 255
        assert isinstance(green, int) and 0 <= green <= 255
        assert isinstance(blue, int) and 0 <= blue <= 255

        self._red = int(red)
        self._green = int(green)
        self._blue = int(blue)

    def rgb(self):
        return self._red, self._green, self._blue

    def bgr(self):
        return self._blue, self._green, self._red

    def __str__(self):
        return "(R={}, G={}, B={})".format(*self.rgb())

class BoundingBox:
    """
    Abstraction for bounding box, including color and line style for visualization
    """

    def __init__(
        self,
        top_left: Point,
        bottom_right: Point,
        confidence: float,
        color=Color(),
        thickness=1,
        label=None,
        label_color=Color(0, 0, 0),
        label_background_color=Color(255, 255, 255),
        label_font_size=1,
        label_transparency=0,
    ):
        # convention: x corresponds to width, y corresponds to height
        assert isinstance(top_left, Point)
        assert isinstance(bottom_right, Point)

        assert bottom_right.x() >= top_left.x()
        assert bottom_right.y() >= top_left.y()
        assert 0 <= confidence <= 1

        self._top_left = top_left
        self._bottom_right = bottom_right
        self._color = color
        self._thickness = thickness
        self._label = label
        self._label_color = label_color
        self._label_background_color = label_background_color
        self._label_font_size = label_font_size
        self._label_transparency = label_transparency
        self._confidence = confidence

    def top_left(self):
        return self._top_left

    def bottom_right(self):
        return self._bottom_right

    def confidence(self):
        return self._confidence

    def height(self):
        return self.bottom_right().y() - self.top_left().y()

    def width(self):
        return self.bottom_right().x() - self.top_left().x()

    def thickness(self):
        return self._thickness

    def label(self):
        return self._label

    def color(self):
        return self._color

    def label_color(self):
        return self._label_color

    def label_background_color(self):
        return self._label_background_color

    def label_font_size(self):
        return self._label_font_size

    def label_transparency(self):
        return self._label_transparency

    def __str__(self):
        return (
            "BoundingBox(height={}, width={}, confidence={}, ".format(
                self.height(), self.width(), self.confidence()
            )
            + "top_left={}, bottom_right={}), ".format(
                self.top_left(), self.bottom_right()
            )
            + "label={}, label_color={}), ".format(self.label(), self.label_color())
            + "label_background_color={}, ".format(self.label_background_color())
            + "label_font_size={})".format(self.label_font_size())
        )

class Frame:
    """
    abstraction class for an image (or video frame)
    """

    def __init__(self, input, HWC=True):
        self._path = None
        self._HWC = HWC

        if isinstance(input, str):
            # input image path
            if not os.path.exists(input):
                logger.warning(f"cannot find input image path: {input}")
                raise RuntimeError(f"cannot find input image path: {input}")
            else:
                # TODO: handle when input is grayscale image
                self._path = input
                self._data = cv2.imread(input)[:, :, ::-1]
        elif isinstance(input, np.ndarray):
            if input.dtype == np.uint8:
                # TODO: handle when input is grayscale, check channel
                self._data = input
            else:
                raise RuntimeError(
                    "input to Frame must be an image path or np.ndarray of type uint8"
                )

    def horizontal_flip(self, inplace=False):
        if inplace:
            self._data = self._data[:, ::-1, :]
        else:
            new_frame = self.copy()
            new_frame.horizontal_flip(inplace=True)
            return new_frame

    def jitter_color(self, inplace=False, hgain=5, sgain=30, vgain=30):
        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)
        img_hsv = cv2.cvtColor(self.data(), cv2.COLOR_RGB2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

        jittered_img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if inplace:
            self._data = jittered_img
        else:
            return Frame(jittered_img)

    def path(self):
        return self._path

    def data(self):
        return self._data

    def height(self):
        if self._HWC:
            # print("4")
            return self._data.shape[0]
        else:
            # Channel of image is not H-W-C (here is CHW)
            return self._data.shape[1]

    def width(self):
        if self._HWC:
            # print("4.")
            return self._data.shape[1]
        else:
            return self._data.shape[2]

    def shape(self):
        return self._data.shape

    def size(self):
        return len(self._data.shape)

    ## TEST
    def update_data(self, newData):
        self._data = newData

    ## TEST
    def to_float(self, new_dtype):
        return

    def to_CHW(self):
        self._data = self._data.transpose(2, 0, 1)

        # TODO: Add bool variable for other code notifying the order has been changed
        self._HWC = False

    ## TODO: Implement this
    def to_HWC(self):
        self._data = self._data.transpose(1, 2, 0)
        self._HWC = True

    ## TODO: center crop function for image classification
    # Ref: https://github.com/pytorch/vision/blob/d010e82fec10422f79c69564de7ff2721d93d278/torchvision/transforms/functional.py#L569
    def center_crop(self, output_size):
        if isinstance(output_size, numbers.Number):
            Hôm
        image_height = self.height()
        # print(f"Image_height:{image_height}")
        image_width = self.width()
        # print(f"Image width: {image_width}")

        output_size = (output_size[0], output_size[0])
        crop_height, crop_width = output_size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2
                if crop_height > image_height
                else 0,
            ]

            pad_left = padding_ltrb[0]
            pad_top = padding_ltrb[1]
            pad_right = padding_ltrb[2]
            pad_bottom = padding_ltrb[3]

            # https://github.com/pytorch/vision/blob/d010e82fec10422f79c69564de7ff2721d93d278/torchvision/transforms/functional_tensor.py#L412
            need_squeeze = False
            if len(self._data.shape) < 4:
                self.data = self._data.unsqueeze(0)
                need_squeeze = True

            self.data = np.pad(self.data, padding_ltrb, mode="constant")

            if need_squeeze:
                self._data = self._data.squeeze(dim=0)

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))

        # Crop part
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height

        if (
            crop_left < 0
            or crop_top < 0
            or crop_right > image_width
            or crop_bottom > image_height
        ):
            padding_ltrb = [
                max(-crop_left + min(0, crop_right), 0),
                max(-crop_top + min(0, crop_bottom), 0),
                max(crop_right - max(image_width, crop_left), 0),
                max(crop_bottom - max(image_height, crop_top), 0),
            ]
            return np.pad(
                self.data[
                    max(crop_top) : crop_bottom, max(crop_left, 0) : crop_right, ...
                ],
                padding_ltrb,
            )

        # print(f"{crop_top}, {crop_bottom},{crop_left},{crop_right}")
        return Frame(self.data()[crop_top:crop_bottom, crop_left:crop_right, ...])

    def get_rescaled_range_0_1_output(self):
        # logger.warning(f"Rescale Frame into [0,1]")

        # logger.info(f"Initial channel order:{self.shape()}")
        # logger.info(f"Change channel order to (C,H,W)")
        self.to_CHW()
        # logger.info(f"Channel after {self.shape()}")

        oldDat = self.data()  # oldDat channel is (C,H,W)
        # logger.info(f"oldDat {np.shape(oldDat)}")
        info = np.iinfo(oldDat.dtype)
        dataNorm = oldDat.astype(np.float64) / info.max

        return dataNorm

    ## Same with convert_image_dtype
    # https://github.com/pytorch/vision/blob/d010e82fec10422f79c69564de7ff2721d93d278/torchvision/transforms/functional_tensor.py#L64
    # and normalized function
    """
    This function will return normalized data from Frame class
    """

    def get_normalized_output(
        self,
        rescaled_input: np.array,
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    ):
        # logger.warning(f"Rescale Frame into [0,1]")

        # logger.info(f"Initial channel order:{self.shape()}")
        # logger.info(f"Change channel order to (C,H,W)")
        # self.to_CHW()
        # logger.info(f"Channel after {self.shape()}")

        # oldDat = self.data()                                            # oldDat channel is (C,H,W)
        # # logger.info(f"oldDat {np.shape(oldDat)}")
        # info = np.iinfo(oldDat.dtype)
        # tempDataNorm = oldDat.astype(np.float64) / info.max

        # data here is already rescaled to [0,1]
        tempDataNorm = rescaled_input
        mean = np.array(mean)
        std = np.array(std)
        tempDataNorm = tempDataNorm - mean.reshape((-1, 1, 1))
        tempDataNorm = tempDataNorm / std.reshape((-1, 1, 1))

        np.transpose(tempDataNorm, (1, 2, 0))  # Return NDarray channel (H,W,C)

        return tempDataNorm
        # return Frame(tmp_norm_data.astype(np.uint8),HWC=False)

    def crop(self, bounding_box: BoundingBox, allow_clipping: bool = False):
        x0, y0 = bounding_box.top_left().x(), bounding_box.top_left().y()
        x1, y1 = bounding_box.bottom_right().x(), bounding_box.bottom_right().y()

        if not allow_clipping:
            if x0 >= 0 and y0 >= 0 and x1 < self.width() and y1 < self.height():
                return Frame(self.data()[y0:y1, x0:x1, :])
            else:
                logger.debug(f"fails to crop frame")
                logger.debug(f"frame info: {self}")
                logger.debug(f"bounding box info: {bounding_box}")
                logger.info(
                    f"if the bounding box exceeds the image size, set allow_clipping to True to crop"
                )
                return None
        else:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(x1, self.width())
            y1 = min(y1, self.height())
            return Frame(self.data()[y0:y1, x0:x1, :])

    def draw_bounding_box(self, box: BoundingBox):
        x0, y0 = box.top_left().x(), box.top_left().y()
        x1, y1 = box.bottom_right().x(), box.bottom_right().y()

        # logger.warning(f"[Dbug]: {x0} {y0} {x1} {y1} {self.width()} {self.height()}")
        if (
            0 <= x0 <= self.width()
            and 0 <= x1 <= self.width()
            and 0 <= y0 <= self.height()
            and 0 <= y1 <= self.height()
        ):
            # valid box
            self._data = draw_rect(
                image=self.data(),
                points=[x0, y0, x1, y1],
                rgb=box.color().rgb(),
                label_transparency=box.label_transparency(),
                thickness=box.thickness(),
                labels=box.label(),
                label_rgb=box.label_color().rgb(),
                label_bg_rgb=box.label_background_color().rgb(),
                label_font_size=box.label_font_size(),
            )
        else:
            logger.warning("got invalid box when putting into the frame")
            logger.warning(f"bounding_box={box}")
            logger.warning(f"frame={self}")
            raise RuntimeError("got invalid box when putting into the frame")

    def draw_bounding_boxes(self, boxes):
        for box in boxes:
            self.draw_bounding_box(box)

    def __str__(self):
        return "Frame(height={}, width={})".format(self.height(), self.width())

    def save(self, path: str):
        ext = path.split(".")[-1]
        assert ext in ["jpg", "png", "JPG", "JPEG", "jpeg"]
        cv2.imwrite(path, self._data[:, :, ::-1])

    def copy(self):
        return Frame(np.copy(self.data()))

    # Idea from: https://github.com/pytorch/vision/blob/d010e82fec10422f79c69564de7ff2721d93d278/torchvision/transforms/functional.py#L366
    def _compute_resized_output_size(self, new_size):
        if len(new_size) == 1:  # specified size only for the smallest edge
            img_height = self.height()
            img_width = self.width()
            short_, long_ = (
                (img_width, img_height)
                if img_width <= img_height
                else (img_height, img_width)
            )
            requested_new_short = new_size if isinstance(new_size, int) else new_size[0]

            new_short_, new_long_ = (
                requested_new_short,
                int(requested_new_short * long_ / short_),
            )

            new_width, new_height = (
                (new_short_, new_long_)
                if img_width <= img_height
                else (new_long_, new_short_)
            )

            # print(f"return as new height: {new_height}, new width: {new_width}")
            return new_height, new_width

    ## My own resize based on Torch for image classification
    def img_class_resize(
        self,
        new_width: int,
        new_height: int,
        keep_ratio: bool = True,
        interpolation="linear",
        hwc: bool = True,
    ):
        img_height = self.height()
        img_width = self.width()

        output_size = [new_height, new_width]

        # print(f"Resizing into new output size {output_size}")

        new_frame_data = cv2.resize(
            self.data(),
            (new_width, new_height),
            interpolation=INTERPOLATIONS[interpolation],
        )

        return Frame(new_frame_data, hwc)

    def resize(
        self,
        new_width: int,
        new_height: int,
        keep_ratio: bool = False,
        pad_constant: int = 0,
        pad_position: str = "fixed",
        interpolation="linear",
        hwc: bool = True,
    ):
        assert interpolation in INTERPOLATIONS
        if keep_ratio:
            # need to keep aspect ratio
            if pad_constant is not None:
                assert isinstance(pad_constant, int)
                assert 0 <= pad_constant <= 255
            else:
                pad_constant = np.random.randint(low=0, high=256)

            assert pad_position in ["fixed", "random"]

            new_frame_data = pad_constant * np.ones(
                (new_height, new_width, 3), dtype=np.uint8
            )
            ratio = min(new_height / self.height(), new_width / self.width())
            sub_width = int(ratio * self.width())
            sub_height = int(ratio * self.height())
            sub_data = cv2.resize(
                self.data(),
                (sub_width, sub_height),
                interpolation=INTERPOLATIONS[interpolation],
            )
            if pad_position == "fixed":
                # put the image to the top left
                new_frame_data[:sub_height, :sub_width, :] = sub_data
            else:
                # randomly put inside
                if new_width - sub_width > 0:
                    start_x = np.random.randint(low=0, high=new_width - sub_width)
                else:
                    start_x = 0
                if new_height - sub_height > 0:
                    start_y = np.random.randint(low=0, high=new_height - sub_height)
                else:
                    start_y = 0

                new_frame_data[
                    start_y : start_y + sub_height, start_x : start_x + sub_width, :
                ] = sub_data

            return Frame(new_frame_data), ratio
        else:
            # no need to keep aspect ratio
            new_frame_data = cv2.resize(
                self.data(),
                (new_width, new_height),
                interpolation=INTERPOLATIONS[interpolation],
            )
            return Frame(new_frame_data, hwc)

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
        logger.info(f"ModelName: {modelName}")
        if modelName in YOLOV5_MODEL_LIST:
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

def ImgClassPreProcess(frame: Frame, config: dict):
    # --------------- ALL IMPORTS GO HERE -------------------------------------
    # -------------------------------------------------------------------------
    import numpy as np

    # --------------- END OF IMPORTS ------------------------------------------
    # -------------------------------------------------------------------------

    # --------------- ALL VAR SETING------------------------------------------
    # -------------------------------------------------------------------------
    config_keys = config.keys()
    isNormalized = False
    isResized = False
    isCenterCroped = False
    # --------------- END OF VAR SETING------------------------------------------
    # -------------------------------------------------------------------------

    meta_data = {
        "height_before": frame.height(),
        "width_before": frame.width(),
    }

    if "resize_size" in config_keys:
        isResized = True
        resize_size = config["resize_size"]
        new_height, new_width = frame._compute_resized_output_size(
            new_size=[resize_size]
        )
        # logger.info(
        #     f"Resizing frame into new_height {new_height} and new_width {new_width}"
        # )
        frame = frame.img_class_resize(
            new_height=new_height, new_width=new_width, keep_ratio=True
        )
        # logger.info(f"After resize size: {frame.shape()}")

    if "crop_size" in config_keys:
        isCenterCroped = True
        # logger.warning(f'Transposing frame into CHW before center crop')
        # frame.to_CHW()

        crop_size = config["crop_size"]
        # logger.info(f"Center crop frame into {crop_size}")
        frame = frame.center_crop([crop_size])
        # logger.info(f"After center crop size: {frame.shape()}")

    if isCenterCroped:
        meta_data["height_after"] = crop_size
        meta_data["width_after"] = crop_size
    elif isResized:
        meta_data["height_after"] = new_height
        meta_data["width_after"] = new_width
    else:
        meta_data["height_after"] = None
        meta_data["width_after"] = None

    if "mean" in config_keys and "std" in config_keys:
        # Call normalized here first
        # Image classification application, all torch models requires scaled to [0,1] before normalized
        rescaled_input = frame.get_rescaled_range_0_1_output()
        isNormalized = True

        # logger.warning(f"Returning normalized NDarray. Not Frame object")
        # logger.info(f"Get normalized ND array")
        mean = config["mean"]
        std = config["std"]
        # logger.info(f"[Before norm] Channel order {frame.shape()}")
        norm_output = frame.get_normalized_output(
            rescaled_input=rescaled_input, mean=mean, std=std
        )  # norm_output shape (C,H,W)

    if isNormalized:
        output = np.ascontiguousarray(norm_output, dtype=np.float32)
        meta_data["mean"] = mean
        meta_data["std"] = std
    else:
        ## TODO: Verify this code since if it is not normalized, then it has be convert to C-H-W ??
        frame.to_CHW()  # Frame here is C-H-W
        output = np.ascontiguousarray(frame.data(), dtype=np.float32)

    ## TODO ? Check if we need to store ratio in resize
    # metadata['resize_ratio'] = ratio

    output = np.expand_dims(output, axis=0)

    # Output return arranged in B-C-H-W
    return output, crop_size, meta_data


def ImaClassPostProcess(model_output, config: dict):
    def softmax(x, axis=None):
        # Subtract the maximum value for numerical stability
        x = x - np.max(x, axis=axis, keepdims=True)
        # Compute the exponential of all elements in the input array
        exp_x = np.exp(x)
        # Compute the sum of exponential values along the specified axis
        sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
        # Divide each element by the sum of exponential values
        return exp_x / sum_exp_x

    class_id_list = config["class_names"]
    prediction = softmax(np.squeeze(model_output[0]))
    # logger.info(prediction.shape)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    categoryName = class_id_list[class_id]

    return categoryName, score


def dummyMaxMin(dummyInputFrame: Frame, config: dict):
    """
    This fuction resize the dummy input into to max_size/min_size from
    model default values for onnx's compatible input.
    Using torchvision transform functional for compatible with torch.onnx.export
    """
    from torchvision.transforms.functional import to_tensor

    config = config["preprocessing"]
    # if "min_size" in config.keys() and "max_size" in config.keys():
    min_size = config["min_size"]
    max_size = config["max_size"]

    resizeDummy = dummyInputFrame.resize(new_height=min_size, new_width=max_size)

    return to_tensor(resizeDummy.data())


def ObjectDetectPreProcess(frame: Frame, config: dict):
    """
    This fucntion only resize the Frame image to max_size/min_size from
    model default values for onnx's compatible input
    """
    # --------------- ALL IMPORTS GO HERE -------------------------------------
    # -------------------------------------------------------------------------
    import numpy as np

    # --------------- END OF IMPORTS ------------------------------------------
    # -------------------------------------------------------------------------

    # --------------- ALL VAR SETING------------------------------------------
    # -------------------------------------------------------------------------
    config_keys = config.keys()
    isRescaled01 = False

    meta_data = {
        "height_before": frame.height(),
        "width_before": frame.width(),
    }

    if "min_size" in config_keys and "max_size" in config_keys:
        min_size = config["min_size"]
        max_size = config["max_size"]
        frame = frame.resize(new_height=min_size, new_width=max_size)

    if "rescaled_01" in config_keys:
        # logger.info("Rescale image to range [0,1]")
        isRescaled01 = True

        # get_rescaled_range_0_1_output will return NDArray with channel (C,H,W)
        rescaled_01_output = frame.get_rescaled_range_0_1_output()

    if isRescaled01:
        meta_data["rescaled_01"] = True

    output = np.ascontiguousarray(rescaled_01_output, dtype=np.float32)
    output = np.expand_dims(output, axis=0)
    # output = np.expand_dims(np.transpose())

    return output.astype(np.float32)


def ObjectDetectPostProcess(frame_batch: Frame, model_output, config: dict):

    # logger.info(config)
    box_coord = model_output[0]

    isLabelLast = config["label_last"]
    if isLabelLast:
        label_arr = model_output[2]
        confidence_arr = model_output[1]
    else:
        label_arr = model_output[1]
        confidence_arr = model_output[2]

    box_coord_shape = np.shape(box_coord)
    box_row = box_coord_shape[0]
    boundingBoxList = []
    frame_batch = np.transpose(frame_batch[0], (1, 2, 0))
    frame_batch *= 255
    # logger.info(frame_batch)
    drawingFrame = Frame(frame_batch.astype(np.uint8), HWC=True)

    # logger.warning(model_output)

    for i in range(box_row):
        coordinate = box_coord[i]
        confidenceVal = confidence_arr[i]
        labelIdx = label_arr[i]
        # logger.info(labelIdx)
        classIdList = config["class_names"]
        label = classIdList[labelIdx]
        # logger.info(label)

        x0, y0, x1, y1 = (
            round(coordinate[0]),
            round(coordinate[1]),
            round(coordinate[2]),
            round(coordinate[3]),
        )
        topLeftPoint = Point(x=x0, y=y0)
        bottomRightPoint = Point(x=x1, y=y1)

        # logger.info(coordinate)
        # logger.info(f"{x0}, {y0}, {x1}, {y1}")
        tmpBoundingBox = BoundingBox(
            top_left=topLeftPoint,
            bottom_right=bottomRightPoint,
            confidence=confidenceVal,
            label=label,
        )
        boundingBoxList.append(tmpBoundingBox)

    drawingFrame.draw_bounding_boxes(boxes=boundingBoxList)

    return drawingFrame.data()


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

    if not os.path.exists(output_dir):
        msg = (
            'Cannot find csv_output" directory.',
            "Please check extractDrive.py again",
        )
        raise RuntimeError(msg)

    isImgClassApplication = False
    isObjDetectApplication = False
    if application == "image_class":
        onnx_model = ImgClassOnnxModel(onnx_path=modelOnnxPathName)
        isImgClassApplication = True

    elif application == "object_detect":
        onnx_model = ObjectDetectOnnxModel(onnx_path=modelOnnxPathName)
        isObjDetectApplication = True
    elif application == "object_detect_custom":
        logger.warning(modelOnnxPathName)
        onnx_model = customObjectDetectOnnxModel(onnx_path=modelOnnxPathName)
        isObjDetectApplication = True

    if isObjDetectApplication:
        drawingResultDir = f"{cwd}/drawingRes/{modelFn}"
        if not os.path.exists(drawingResultDir):
            logger.warning(
                f"Creating drawing object detection directory {drawingResultDir}..."
            )
            os.mkdir(drawingResultDir)
    
    # WARM UP MODEL FOR CACHE LOAD
    for i in range(10):
        
        # using random input for object detection torchvision models (high version) will yield error
        # onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. Name:'/roi_heads/Reshape' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/tensor/reshape_helper.h:36 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, onnxruntime::TensorShapeVector&, bool) size != 0 && (input_shape.Size() % size) == 0 was false. The input tensor cannot be reshaped to the requested shape. Input shape:{0,364}, requested shape:{0,-1}

        # randomDump = np.random.rand(500,500,3)
        # randomDump = Frame(randomDump.astype(np.uint8))

        # assert isImgClassApplication is True or isObjDetectApplication is True
        # if isImgClassApplication:
        #     classOut, scoreOut = onnx_model(randomDump)
        # elif isObjDetectApplication:
        #     drawingFrameOut = onnx_model(randomDump)

        imgIdx = os.path.join(COCO_verify_dir, f"{i}.jpg")
        tempImg = Frame(imgIdx)

        assert isImgClassApplication is True or isObjDetectApplication is True
        if isImgClassApplication:
            classOut, scoreOut = onnx_model(tempImg)
        elif isObjDetectApplication:
            drawingFrameOut = onnx_model(tempImg)
    
    logger.info("Finish warm up model. Sleep for 5 minutes before benchmarking...")
    t.sleep(timeSleep)

    scoreClasstDict = {}
    timeBenchmarkList = []
    engineLoadTime = onnx_model.engineTime
    for i in tqdm(range(numIteration)):
        imgIdx = os.path.join(COCO_verify_dir, f"{i}.jpg")
        tempImg = Frame(imgIdx)

        assert isImgClassApplication is True or isObjDetectApplication is True
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
    output_dir = os.path.join(output_dir, csvFileOut)
    # timeBenchmarkDF = pd.DataFrame(timeBenchmarkArr)
    # timeBenchmarkDF.to_csv(output_dir, header=False, index=False)
    np.savetxt(output_dir, timeBenchmarkArr, delimiter=",")
    print(f"Finish exporting result to file {csvFileOut}. Sleeping for 5 mins...")

    t.sleep(timeSleep)