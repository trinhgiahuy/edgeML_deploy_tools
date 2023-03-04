import sys
import os
import argparse
import numpy as np

from loguru import logger
import tarfile
import dill
import matplotlib.pyplot as plt

from torchvision.io.image import read_image
from common_cvinfer import *

from google.protobuf.json_format import MessageToDict
import torchvision.transforms as T
import json

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f"Please install onnx and onnxruntime first. {e}")


def is_set(arg_name):
    if arg_name in sys.argv:
        return True
    return False


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
    # logger.warning(f"Frame before {frame.shape()}")

    meta_data = {
        "height_before": frame.height(),
        "width_before": frame.width(),
    }
    # logger.warning(frame.shape())
    if "new_height" in config_keys and "new_width" in config_keys:
        # logger.warning("1")

        new_height = config["new_height"]
        new_width = config["new_width"]
        frame = frame.resize(
            new_width=new_width,
            new_height=new_height,
        )

        frameToDraw = frame
        meta_data["height_after"] = frame.height()
        meta_data["width_after"] = frame.width()

    if "min_size" in config_keys and "max_size" in config_keys:
        min_size = config["min_size"]
        max_size = config["max_size"]
        frame = frame.resize(new_height=min_size, new_width=max_size)

    if "rescaled_01" in config_keys:
        # logger.info("Rescale image to range [0,1]")
        isRescaled01 = True

        # get_rescaled_range_0_1_output will return NDArray with channel (C,H,W)
        rescaled_01_output = frame.get_rescaled_range_0_1_output()
        output = rescaled_01_output

    else:
        # # For custom model like tinyYOLOv2
        output = frame.data()
        output = np.transpose(output, (2, 0, 1))

    if isRescaled01:
        meta_data["rescaled_01"] = True

    output = np.ascontiguousarray(output, dtype=np.float32)
    output = np.expand_dims(output, axis=0)
    # logger.info(np.shape(output))

    return output, meta_data, frameToDraw


# def old(onnx_output, img):
#     """
#     args:
#         @imgFrameData: must be resized Frame and shape (C,H,W)
#     """

#     numClasses = 20
#     anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

#     def sigmoid(x, derivative=False):
#         return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

#     def softmax(x):
#         scoreMatExp = np.exp(np.asarray(x))
#         return scoreMatExp / scoreMatExp.sum(0)

#     clut = [
#         (0, 0, 0),
#         (255, 0, 0),
#         (255, 0, 255),
#         (0, 0, 255),
#         (0, 255, 0),
#         (0, 255, 128),
#         (128, 255, 0),
#         (128, 128, 0),
#         (0, 128, 255),
#         (128, 0, 128),
#         (255, 0, 128),
#         (128, 0, 255),
#         (255, 128, 128),
#         (128, 255, 128),
#         (255, 255, 0),
#         (255, 128, 128),
#         (128, 128, 255),
#         (255, 128, 128),
#         (128, 255, 128),
#         (128, 128, 128),
#     ]

#     label = [
#         "aeroplane",
#         "bicycle",
#         "bird",
#         "boat",
#         "bottle",
#         "bus",
#         "car",
#         "cat",
#         "chair",
#         "cow",
#         "diningtable",
#         "dog",
#         "horse",
#         "motorbike",
#         "person",
#         "pottedplant",
#         "sheep",
#         "sofa",
#         "train",
#         "tvmonitor",
#     ]

#     blockSize = 32
#     gridWidth = int(416 / blockSize)
#     gridHeight = int(416 / blockSize)
#     draw = ImageDraw.Draw(img)
#     for cy in range(gridHeight):
#         for cx in range(gridWidth):
#             for b in range(5):
#                 channel = b * (numClasses + 5)
#                 tx = onnx_output[channel][cy][cx]
#                 ty = onnx_output[channel + 1][cy][cx]
#                 tw = onnx_output[channel + 2][cy][cx]
#                 th = onnx_output[channel + 3][cy][cx]
#                 tc = onnx_output[channel + 4][cy][cx]
#                 x = (float(cx) + sigmoid(tx)) * 32
#                 y = (float(cy) + sigmoid(ty)) * 32

#                 w = np.exp(tw) * 32 * anchors[2 * b]
#                 h = np.exp(th) * 32 * anchors[2 * b + 1]

#                 confidence = sigmoid(tc)

#                 classes = np.zeros(numClasses)
#                 for c in range(0, numClasses):
#                     classes[c] = onnx_output[channel + 5 + c][cy][cx]
#                 classes = softmax(classes)
#                 detectedClass = classes.argmax()

#                 if 0.5 < classes[detectedClass] * confidence:
#                     logger.info(detectedClass)
#                     color = clut[detectedClass]
#                     x = x - w / 2
#                     y = y - h / 2

#                     x0, y0, x1, y1 = (round(x), round(y), round(x + w), round(y + h))
#                     logger.info(f"{x0} {y0} {x1} {y1}")
#                     topLeftPoint = Point(x=x0, y=y0)
#                     bottomRightPoint = Point(x=x1, y=y1)
#                     tmpBoundingBox = BoundingBox(
#                         top_left=topLeftPoint,
#                         bottom_right=bottomRightPoint,
#                         confidence=classes[detectedClass] * confidence,
#                         # label=label,
#                     )
#                     boundingBoxList.append(tmpBoundingBox)

#                     draw.line((x, y, x + w, y), fill=color)
#                     draw.line((x, y, x, y + h), fill=color)
#                     draw.line((x + w, y, x + w, y + h), fill=color)
#                     draw.line((x, y + h, x + w, y + h), fill=color)

#     return img
#     # img.save("result.png")


def ObjectDetectPostProcess(onnx_output, imgFrameData: Frame, config: dict):
    """
    args:
        @imgFrameData: must be resized Frame and shape (H,W,C)
        @config: postprocessing configuration dictionary
    """

    numClasses = 20

    def sigmoid(x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    def softmax(x):
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

    blockSize = 32
    gridWidth = int(416 / blockSize)
    # gridWidth = 13
    gridHeight = int(416 / blockSize)
    # gridHeight = 13

    drawingFrame = imgFrameData
    label_classes = config["class_names"]
    confidence_threshold = config["score_thresh"]
    anchors = config["anchors"]

    boundingBoxList = []
    for cy in range(gridHeight):
        for cx in range(gridWidth):
            for b in range(5):
                channel = b * (numClasses + 5)
                tx = onnx_output[channel][cy][cx]
                ty = onnx_output[channel + 1][cy][cx]
                tw = onnx_output[channel + 2][cy][cx]
                th = onnx_output[channel + 3][cy][cx]
                tc = onnx_output[channel + 4][cy][cx]
                x = (float(cx) + sigmoid(tx)) * 32
                y = (float(cy) + sigmoid(ty)) * 32

                w = np.exp(tw) * 32 * anchors[2 * b]
                h = np.exp(th) * 32 * anchors[2 * b + 1]

                confidence = sigmoid(tc)

                classes = np.zeros(numClasses)
                for c in range(0, numClasses):
                    classes[c] = onnx_output[channel + 5 + c][cy][cx]
                classes = softmax(classes)
                detectedClass = classes.argmax()

                if classes[detectedClass] * confidence > confidence_threshold:
                    x = x - w / 2
                    y = y - h / 2
                    x0, y0, x1, y1 = (round(x), round(y), round(x + w), round(y + h))

                    x0 = 0 if x0 < 0 else x0
                    y0 = 0 if y0 < 0 else y0
                    x1 = drawingFrame.width() if x1 > drawingFrame.width() else x1
                    y1 = drawingFrame.height() if y1 > drawingFrame.height() else y1

                    label = label_classes[detectedClass]
                    topLeftPoint = Point(x=x0, y=y0)
                    bottomRightPoint = Point(x=x1, y=y1)
                    tmpBoundingBox = BoundingBox(
                        top_left=topLeftPoint,
                        bottom_right=bottomRightPoint,
                        confidence=classes[detectedClass] * confidence,
                        label=label,
                    )
                    boundingBoxList.append(tmpBoundingBox)

    drawingFrame.draw_bounding_boxes(boxes=boundingBoxList)

    return drawingFrame.data()


def write_artifacts(
    prefix: str, pytorch_model, config: dict, applicationType: str, test_frame: Frame
):
    """
    This function is used to write the artifacts
    Args:
        @config: model config dictionary contain preprocessing, postprocessing keys
    """

    # -------------- SERIALIZE THE PRE/POSTPROCESSING FUNCTIONS -----------
    if applicationType == "ImageClassification":
        with open(prefix + ".preprocess", "wb") as fid:
            dill.dump(ImgClassPreProcess, fid)

        with open(prefix + ".postprocess", "wb") as fid:
            dill.dump(ImaClassPostProcess, fid)

    elif applicationType == "ObjectDetection":
        with open(prefix + ".preprocess", "wb") as fid:
            dill.dump(ObjectDetectPreProcess, fid)

        with open(prefix + ".postprocess", "wb") as fid:
            dill.dump(ObjectDetectPostProcess, fid)

    # ------------- END OF SERIALIZING PRE/POSTPROCESSING FUNCTIONS -------

    # -------------- WRITE DEPENDENCIES ------------------------------
    DEPENDENCIES = [
        "numpy",
        "dill",
        "tqdm",
        "pandas",
        "scikit-image",
        "onnxruntime",
        "loguru",
        "gdownload",
        "drawline",
    ]

    logger.warning("Dependencies are the following")
    logger.warning(f"{DEPENDENCIES}")

    with open(prefix + ".dependencies.txt", "w") as fid:
        for item in DEPENDENCIES:
            fid.write(item + "\n")
    # -------------- END OF WRITING DEPENDENCIES --------------------

    # -------------- WRITE SAMPLE CONFIG ----------------------------
    CONFIG = modelConf
    logger.warning("Writting model configuration into json config file")
    json.dumps(CONFIG, indent=2)

    config_file = prefix + ".configuration.json"
    with open(config_file, "w") as fid:
        fid.write(json.dumps(CONFIG, indent=2))

    logger.info(f"complete writing sample configurations to {config_file}")
    # -------------- END OF WRITING SAMPLE CONFIG -------------------

    # -------------- GET DUMMY INPUT BASED ON APPLICATION------------

    if applicationType == "ObjectDetection":
        # Here we need dummy for onnx model input compatible
        dummy_input = dummyMaxMin(dummyInputFrame=test_frame, config=config)
        dummy_input = dummy_input.unsqueeze(0)

    # ---------------------------------------------------------------

    # -------------- CONVERT PYTORCH/TF/MXNET MODEL TO ONNX ---------
    # ---------------------------------------------------------------


def printVerify(onnxRes, ptRes):
    print(onnxRes)
    print()
    print(ptRes)


def zip_artifact(prefix):
    logger.warning("Start ziping exported artifact files")
    tarFn = prefix + ".onnx.tar.gz"
    tar = tarfile.open(tarFn, "w:gz")

    for file in [
        prefix + ".preprocess",
        prefix + ".onnx",
        prefix + ".postprocess",
        prefix + ".configuration.json",
        prefix + ".dependencies.txt",
    ]:
        tar.add(file, arcname=os.path.basename(file).split("/")[-1])

    tar.close()
    logger.warning("Finish zipping")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        required=True,
        dest="input_file",
        help="Input Torch Model File",
    )
    parser = args.parse_args()
    # logger.warning(parser.input_file)

    if is_set(parser.input_file):
        # print(parser.input_file)
        INPUT_MODEL_NAME = parser.input_file

    # Load config params from JSON config file (modelConfig.py)
    from modelConfig import *

    # Flag for controlling application
    isImgClassApplication = False
    isObjDetectApplication = False

    if INPUT_MODEL_NAME in imgClassModelsCfg.keys():
        modelConf = imgClassModelsCfg[INPUT_MODEL_NAME]
        isImgClassApplication = True

    if INPUT_MODEL_NAME in objDetModelsCfg.keys():
        modelConf = objDetModelsCfg[INPUT_MODEL_NAME]
        isObjDetectApplication = True

    preProcessing = modelConf["preprocessing"]
    postProcessing = modelConf["postprocessing"]

    img = Frame("/home/user/datTran/submission/data/Grace_Hopper.jpg", HWC=True)

    pil_img = read_image("data/Grace_Hopper.jpg")
    pil_img2 = read_image("data/desk.jpg")

    frame_img = Frame("/home/user/datTran/submission/data/Grace_Hopper.jpg")
    frame_img2 = Frame("/home/user/datTran/submission/data/desk.jpg")

    if "custom" in preProcessing.keys():
        logger.info("Using custom image")
        model_onnx_path = customOnnxModel[INPUT_MODEL_NAME]

    # Try inference first
    onnx_model = onnx.load(model_onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = rt.InferenceSession(model_onnx_path)
    input_name = ort_session.get_inputs()[0].name

    inputAll = [node.name for node in onnx_model.graph.input]
    inputInitializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(inputAll) - set(inputInitializer))
    assert len(net_feed_input) == 1

    for i in range(20):
        # for i in [6,7,8,9,10,11,12,13,14,15,16,111,222,333]:
        batchName = "5000_COCO_imgs/" + str(i) + ".jpg"

        batchDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), batchName)
        frame_batch = Frame(batchDir)
        batchTmp, _, frameToDraw = ObjectDetectPreProcess(
            frame=frame_batch, config=preProcessing
        )

        # Here my code works
        # new_height = preProcessing['new_height']
        # new_width = preProcessing['new_width']
        # frame_resized = frame_batch.resize(new_height=new_height,new_width=new_width)
        # frame_resized_cp = frame_resized
        # frame_resized_cp.to_CHW()
        # output = frame_resized_cp.data()
        # output = np.ascontiguousarray(output, dtype=np.float32)
        # output = np.expand_dims(output, axis=0)

        # # @batchTmp (float32 1,3,416,416)
        # logger.info(np.shape(batchTmp))

        # ORIGINAL REF CODE
        # img = Image.open(batchDir)
        # img = img.resize((416, 416))
        # frame_batch_org = np.asarray(img)
        # frame_batch_org = frame_batch_org.transpose(2, 0, 1)
        # frame_batch_org = frame_batch_org.reshape(1, 3, 416, 416).astype(np.float32)
        # logger.warning(f"Org: {frame_batch_org}")
        # logger.warning(f"My: {batchTmp}")

        onnxTmp = ort_session.run(None, {input_name: batchTmp})
        drawingRet = ObjectDetectPostProcess(
            onnx_output=onnxTmp[0][0], imgFrameData=frameToDraw, config=postProcessing
        )
        # drawingRet = old(onnx_output=onnxTmp[0][0],img=img)

        plt.figure()
        plt.imshow(drawingRet)
        plt.show()


"""
    if isImgClassApplication:
        applicationType = "ImageClassification"
    elif isObjDetectApplication:
        applicationType = "ObjectDetection"

    else:
        print("Unknown input file. Cannot find in database")


    import platform
    python_ver = platform.python_version()
    output_dir = f"./onnx_output_{python_ver}/"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    prefix = os.path.join(output_dir, INPUT_MODEL_NAME)
    write_artifacts(
        prefix=prefix,
        pytorch_model=model,
        config=modelConf,
        applicationType=applicationType,
        test_frame=frame_img,
    )

    logger.info(
        f"Finish writting model {INPUT_MODEL_NAME}'s artifact files to {prefix}"
    )
    zip_artifact(prefix=prefix)
"""
