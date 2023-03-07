#!/bin/bash

# SETTING FOR EXECUTION
numIteration=5000

application=$1
prefix=$2

#  Down load CIFAR dataset
# echo "Downloading CIFAR dataset..."
# python -c 'from extractDrive import *; print(downloadCIFAR())'

# Download COCO dataset
python3.6 -c 'from  extractDrive import *; print(downloadCOCO())'


# echo $arrModelList
# echo "Executing extract models from Google Drive..."
# python -c 'from utils import *; print(downloadImageClass())'

echo "Download all models for application ${application}"

python3.6 extractDrive.py --application $application --prefix $prefix

if [ "$application" = "image_class" ]; then
    arrModelList=`python3.6 -c 'from constant import *; print(" ".join(getImageClassModelList()))'`
fi

if [ "$application" = "object_detect" ]; then
    arrModelList=`python3.6 -c 'from constant import *; print(" ".join(getObjectDetectModelList()))'`
fi

if [ "$application" = "object_detect_custom" ]; then
    arrModelList=`python3.6 -c 'from constant import *; print(" ".join(getObjectDetectCustomModelList()))'`
fi


echo $arrModelList

for modelName in $arrModelList
do
    modelFn="${modelName}"
    echo "Executing classifier onnx model ${modelFn}"
    python3.6 onnxClassifier.py --prefix $prefix --application $application --modelFn $modelFn --numIteration $numIteration
done