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


def downloadApplicationsModel(name: str, application: str, prefix: str):
    applicationModelURL = modelLinkDrive[name]

    # For jetson Nano using this line
    # applicationModelURL = JetsonNanoLinkDrive[name]

    # For Jetson Tx2 using this line
    # applicationModelURL = JetsonTX2LinkDrive[name]

    logger.info(f"Extracting model: {name} ... from {applicationModelURL}")

    appPath = f"{os.getcwd()}/{application}_{prefix}"
    appTar = name + f".{prefix}.tar.gz"

    if not os.path.exists(appPath):
        logger.warning(f"Directory {appPath} not found. Creating...")
        os.mkdir(appPath)
    else:
        logger.warning(f"Directory {appPath} already exists!!")

    appFile = os.path.join(appPath, appTar)
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


if __name__ == "__main__":
    if not os.path.exists(csv_output_dir):
        logger.warning(f"Directory {csv_output_dir} not found. Creating...")
        os.mkdir(csv_output_dir)
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
    args = parser.parse_args()

    application = args.application
    # application = "object_detect"

    prefix = args.prefix

    if application == "image_class":
        for name in imageClassModelName:
            downloadApplicationsModel(name=name, application=application, prefix=prefix)

    if application == "object_detect":
        for name in objectDetectModelName:
            downloadApplicationsModel(name=name, application=application, prefix=prefix)

    if application == "object_detect_custom":
        for name in customobjectDetectModelName:
            downloadApplicationsModel(name=name, application=application, prefix=prefix)
