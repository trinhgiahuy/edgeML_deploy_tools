#!/bin/bash

# SETTING FOR EXECUTION
numIteration=5000

application=$1
prefix=trt
isJetson=True

echo "Download all models for application ${application}"

python extractDrive.py --application $application --prefix $prefix --isJetson $isJetson

if [ "$application" = "image_class" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getImageClassModelList()))'`
fi

if [ "$application" = "object_detect" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getObjectDetectModelList()))'`
fi

if [ "$application" = "object_detect_custom" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getObjectDetectCustomModelList()))'`
fi

if [ "$application" = "object_detect_yolox" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getYOLOXObjectDetectModelList()))'`
fi

if [ "$application" = "human_pose" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getHumanPoseModelList()))'`
fi

if [ "$application" = "seman_segmen" ]; then
     arrModelList=`python -c 'from constant import *; print(" ".join(getSemanSegmenModelList()))'`
fi

echo $arrModelList
echo "Building engine for all models in ${application}"


# Uncomment this if already have engine file in GG Drive
# for modelName in $arrModelList
# do
#     modelFn="${modelName}"
#     python buildEngine.py --application $application --modelFn $modelFn --prefix $prefix
# done


# Uncomment this if using above buiding block code
for modelName in $arrModelList
do
    modelFn="${modelName}"
    echo "Executing classifier tensorRT model ${modelFn}"
    python deployTensorRT.py --prefix $prefix --application $application --modelFn $modelFn --numIteration $numIteration
done