import os
import tarfile
from constant import *
from loguru import logger
import gdown
import argparse

cwd = os.getcwd()
COCODir = cwd + "/COCO_5000_imgs/"
COCOExist = os.path.exists(COCODir)
csv_output_dir = f"{cwd}/csv_output"
csvOutDirCache = f"{cwd}/cache"
csvOutDirNoCache = f"{cwd}/no_cache"
drawing_detect_dir = f"{cwd}/drawingRes"


def downloadCOCO():
    if not COCOExist:
        logger.info("COCO image directory is not exist. Creating...")
        os.mkdir(COCODir)
        COCOTarFile = "5000_COCO_imgs.tar.gz"
        COCOZipFile = os.path.join(COCODir, COCOTarFile)
        if not os.path.isfile(COCOZipFile):
            logger.info("Downloading COCO zip file...")
            gdown.download(COCODriveLink, COCOZipFile, quiet=False)

            logger.info("Extracting COCO image dataset...")
            COCOFile = tarfile.open(COCOZipFile)
            COCOFile.extractall(COCODir)
            COCOFile.close()
        else:
            logger.info("File already exist!")
    else:
        logger.info("COCO Image Dataset already exists")


def downloadApplicationsModel(name: str, application: str, prefix: str, isJetson: bool):

    if isJetson:
        if prefix == "trt":
            applicationModelURL = jetson36ModelLink[name]
        elif prefix == "onnx":
            # For Jetson Tx2 using this line
            applicationModelURL = JetsonTX2LinkDrive[name]
        else:
            logger.warning(f"Unknown prefix {prefix}")
    else:
        # For pi, NEVER REACH HERE
        applicationModelURL = modelLinkDrive[name]
    # Unused this since Jetson Nano upgraded to use Python 3.8
    # For jetson Nano using this line
    # applicationModelURL = JetsonNanoLinkDrive[name]


    logger.info(f"Extracting model: {name} ... from {applicationModelURL}")

    appPath = f"{os.getcwd()}/{application}_{prefix}"
    appTar = name + f".{prefix}.tar.gz"

    if not os.path.exists(appPath):
        logger.warning(f"Directory {appPath} not found. Creating...")
        os.mkdir(appPath)
    else:
        logger.warning(f"Directory {appPath} already exists!!")

    appFile = os.path.join(appPath, appTar)

    if prefix == "onnx":
        onnxAppFile = appFile.replace(".onnx.tar.gz", ".onnx")

        if not os.path.isfile(appFile):
            logger.warning(f"Application model {name} not found. Downloading...")
            gdown.download(applicationModelURL, appFile, quiet=False)
        else:
            logger.warning(f"Application model {name} found.")

        if not os.path.isfile(onnxAppFile):
            logger.warning(f"File is not extracted.Extracting application model {name}...")
            appFileTmp = tarfile.open(appFile)
            appFileTmp.extractall(appPath)
            appFileTmp.close()
            logger.info(f"Finish extracting application model{name}")
    elif prefix == "trt":
        trtAppFile = appFile.replace(".trt.tar.gz", ".trt")
        if not os.path.isfile(appFile):
            logger.warning(f"Application model {name} not found. Downloading...")
            gdown.download(applicationModelURL, appFile, quiet=False)
        else:
            logger.warning(f"Application model {name} found.")

        # print(appFile)
        # print(trtAppFile)
        # print(appPath)
        if not os.path.isfile(trtAppFile):
            logger.warning(f"File is not extracted.Extracting application model {name}...")
            appFileTmp = tarfile.open(appFile)
            appFileTmp.extractall(appPath)
            appFileTmp.close()
            logger.info(f"Finish extracting application model{name}")

if __name__ == "__main__":
    # if not os.path.exists(csv_output_dir):
    #     logger.warning(f"Directory {csv_output_dir} not found. Creating...")
    #     os.mkdir(csv_output_dir)
    if not os.path.exists(csvOutDirCache):
        logger.warning(f"Directory {csvOutDirCache} not found. Creating")
        os.mkdir(csvOutDirCache)

    if not os.path.exists(csvOutDirNoCache):
        logger.warning(f"Directory {csvOutDirNoCache} not found. Creating")
        os.mkdir(csvOutDirNoCache)

    if not os.path.exists(drawing_detect_dir):
        logger.warning(f"Directory {drawing_detect_dir} not found. Creating...")
        os.mkdir(drawing_detect_dir)

    downloadCOCO()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--application", type=str, required=True, help="(image_class,object_detect)"
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="(onnx, OpenVINO, tvn)"
    )
    parser.add_argument(
        "--isJetson", type=bool, required=True, help="isJetsonOrNot"
    )

    args = parser.parse_args()

    application = args.application
    prefix = args.prefix
    isJetson = args.isJetson

    if application == "image_class":
        for name in imageClassModelName:
            downloadApplicationsModel(name=name, application=application, prefix=prefix, isJetson=isJetson)

    if application == "object_detect":
        for name in objectDetectModelName:
            downloadApplicationsModel(name=name, application=application, prefix=prefix, isJetson=isJetson)

    if application == "object_detect_custom":
        for name in customobjectDetectModelName:
            downloadApplicationsModel(name=name, application=application, prefix=prefix, isJetson=isJetson)

    if application == "object_detect_yolox":
        for name in YOLOXObjectDetectOnnxModelName:
            logger.warning(f"Get name {name}")
            downloadApplicationsModel(name=name, application=application, prefix=prefix, isJetson=isJetson)

    if application == "human_pose":
        for name in HumanPoseOnnxModelName:
            logger.warning(f"Get name {name}")
            downloadApplicationsModel(name=name, application=application, prefix=prefix, isJetson=isJetson)
    
    if application == "seman_segmen":
        for name in SemanSegmenModelName:
            logger.warning(f"Get name {name}")
            downloadApplicationsModel(name=name, application=application, prefix=prefix, isJetson=isJetson)