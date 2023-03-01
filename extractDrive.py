import os
import tarfile
import requests
from constant import *
from loguru import logger
import gdown

cwd = os.getcwd()
COCODir = cwd + "/COCO_500_imgs/"
COCOExist = os.path.exists(COCODir)


def downloadCOCO():
    if not COCOExist:
        logger.info("COCO image directory is not exist. Creating...")
        os.mkdir(COCODir)

    else:
        logger.info("COCO Image Dataset already exists")

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


def downloadApplicationsModels(application: str, prefix: str):
    for name in imageClassModelName:
        applicationModelURL = modelLinkDrive[name]
        logger.info(f"Extracting model: {name} ... from {applicationModelURL}")

        appPath = f"{os.getcwd()}/{application}_{prefix}"
        appTar = name + f".{prefix}.tar.gz"

        if not os.path.exists(appPath):
            logger.warning(f"Directory {appPath} not found. Creating...")
            os.mkdir(appPath)
        else:
            logger.warning(f"Directory {appPath} already exists!!")

        appFile = os.path.join(appPath, appTar)
        # logger.info(f"appFile: {appFile}")

        if not os.path.isfile(appFile):
            logger.warning(f"Application model {name} not found. Downloading...")
            gdown.download(applicationModelURL, appFile, quiet=False)
        else:
            logger.warning(f"Application model {name} found.")

        logger.info(f"Extracting application model {name}...")
        appFileTmp = tarfile.open(appFile)
        appFileTmp.extractall(appPath)
        appFileTmp.close()
        logger.info(f"Finish extracting application model{name}")


if __name__ == "__main__":
    # downloadCOCO()
    application = "img_class"
    prefix = "onnx"
    downloadApplicationsModels(application=application, prefix=prefix)
