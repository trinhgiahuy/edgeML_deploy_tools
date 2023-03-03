#!/bin/bash

application=$1

if [ "$application" = "image_class" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getImageClassModelList()))'`
fi

if [ "$application" = "object_detect" ]; then
    arrModelList=`python -c 'from constant import *; print(" ".join(getObjectDetectModelList()))'`
fi

for name in $arrModelList
do
	echo "CSV output of ${name}"
	awk -F"," '{pre+=$1;inf+=$2;post+=$3}END{print "pre " pre " inf " inf " post " post}' ./csv_output/"${name}_onnx.csv"
done
