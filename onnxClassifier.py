import os
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
from time import time

from common_cvinfer import *

try:
    import onnx
    import onnxruntime as rt
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
        help="ML application (img_class,obj_det)",
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
    COCO_verify_dir = f"{cwd}/COCO_500_imgs"
    modelDir = f"{cwd}/img_class_{prefix}"
    modelOnnxPathName = os.path.join(modelDir, modelFn + ".onnx")
    logger.info(modelOnnxPathName)
    if not os.path.exists(COCO_verify_dir):
        msg = (
            'Cannot find "COCO_500_imgs" directory.',
            "Please download COCO image dataset first",
        )
        raise RuntimeError(msg)

    if application == "img_class":
        onnx_model = ImgClassOnnxModel(onnx_path=modelOnnxPathName)
    elif application == "obj_det":
        onnx_model = ObjectDetectOnnxModel(onnx_path=modelOnnxPathName)

    scoreClasstDict = {}
    timeBenchmarkList = []
    engineLoadTime = onnx_model.getEngineTime()
    for i in tqdm(range(numIteration)):
        imgIdx = os.path.join(COCO_verify_dir, f"{i}.jpg")
        tempImg = Frame(imgIdx)
        classOut, scoreOut = onnx_model(tempImg)
        preProcessTime, inferenceTime, postProcessTime = (
            onnx_model.preProcessTime,
            onnx_model.inferenceTime,
            onnx_model.postProcessTime,
        )
        scoreClasstDict[classOut] = scoreOut
        tempTimeList = [preProcessTime, inferenceTime, postProcessTime]
        timeBenchmarkList.append(tempTimeList)

    logger.info("Finish benchmarking!!")
    timeBenchmarkArr = np.array(timeBenchmarkList)
    logger.info(timeBenchmarkArr)
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
    timeBenchmarkDF = pd.DataFrame(timeBenchmarkArr)
    timeBenchmarkDF.to_csv(csvFileOut, header=False, index=False)
    print(f"Finish exporting result to file {csvFileOut}")
