import os
from tqdm import tqdm
import numpy as np
import argparse

# import pandas as pd
import time as t
from time import time

from skimage.io import imsave
from common_cvinfer import *

try:
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f"Please install onnx and onnxruntime first. {e}")


if __name__ == "__main__":

    timeSleep = 5*60

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
    for i in range(100):
        
        randomDump = np.random.rand(500,500,3)
        randomDump = Frame(randomDump.astype(np.uint8))

        assert isImgClassApplication is True or isObjDetectApplication is True
        if isImgClassApplication:
            classOut, scoreOut = onnx_model(randomDump)
        elif isObjDetectApplication:
            drawingFrameOut = onnx_model(randomDump)
    
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
            imsave(imgPath, drawingFrameOut)
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

