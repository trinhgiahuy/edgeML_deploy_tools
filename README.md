## edgeML_deploy_tools

Devices' IP address
- Raspberry Pi 3B : `pi@192.168.13`
- Raspberry Pi 4 : `pi@192.168.1.23`
- Jetson Nano : `proe@192.168.1.12`
- Jetson TX2 : `ubuntu@192.168.1.17`
```sh
Python 3.6.9 
CUDA 10.2 
onnx==1.11.0 
onnxruntime-gpu @ file:///home/ubuntu/edge_software/edge_software/edge_sw/bootstrap/onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
tensorrt==7.1.3.0
```

Check cudnn from [this](https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation)

Fundamental tools to run inference on edge devices

## Usage Steps:

### Clone github repository

```sh
git clone 
```

### Convert all models to specified target 

- For example 

```sh
bash ./convertAll.sh image_class
```

### Execute inference

Example to run

For Jetson Nano (Python 3.8.16 version), in `extractDrive.py` using this line `applicationModelURL = JetsonNanoLinkDrive[name]` instead

Execute run

```sh
bash ./execute.sh object_detect onnx
```

Image classification command example

```sh
python onnxClassifier.py --numIteration 2 --modelFn alexnet --application img_class --prefix onnx
```

Object detection command example

```sh
python onnxClassifier.py --numIteration 2 --modelFn fasterrcnn_resnet50_fpn_v2 --application object_detect --prefix onnx
```

### Models to be run on devices

JetsonTX2=[
    "alexnet",
    "googlenet",
    "efficientnet_b0",
    "densenet121",
    "squeezenet1_0",
    "inception_v3",
    "shufflenet_v2_x0_5",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0"
    "mnasnet1_3",
    "resnet18"
]
