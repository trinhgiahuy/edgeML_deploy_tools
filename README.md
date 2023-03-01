## edgeML_deploy_tools

Fundamental tools to run inference on edge devices

Example to run

Image classification command example

```sh
python onnxClassifier.py --numIteration 2 --modelFn alexnet --application img_class --prefix onnx
```

Object detection command example

```sh
python onnxClassifier.py --numIteration 2 --modelFn fasterrcnn_resnet50_fpn_v2 --application object_detect --prefix onnx
```
