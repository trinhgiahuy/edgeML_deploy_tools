COCODriveLink = "https://drive.google.com/u/2/uc?id=1gE5C1UXcCyogZU3K95tn-bXfuF8a-8h4"

imageClassModelName = [
    "alexnet",
    # # "resnet18",
    # "resnet50",
    # "resnet101",
    # "resnet152",
    # "resnext50_32x4d",
    # "resnext101_32x8d",
    # "resnext101_64x4d",
    # "googlenet",
]

objectDetectModelName = [
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn",
    "fcos_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
]

modelLinkDrive = {
    "alexnet": "https://drive.google.com/u/2/uc?id=1tv9NPj9RaYAsek8hCnsFx_YPdfyt9c6t",
    "fasterrcnn_resnet50_fpn_v2": "https://drive.google.com/u/2/uc?id=1W3w80Jloy1WhnNKYi9xRmKQeeWhh9Yyr",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "https://drive.google.com/u/2/uc?id=19OOriiJ4dVsyDBhKNj05xk9KFdlx0-Dl",
    "fasterrcnn_mobilenet_v3_large_fpn": "https://drive.google.com/u/2/uc?id=1zQbGNWu8xl2DbmX0GyUc2gKrOpr2N89G",
    "fasterrcnn_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1vJm13Ay5FEgSVkYniBA5jS7jihvH7KNy",
    "fcos_resnet50_fpn": "https://drive.google.com/u/2/uc?id=17WQn3fmkPrYNRNlEWuqLdXuaBM_QTy9f",
    "retinanet_resnet50_fpn_v2": "https://drive.google.com/u/2/uc?id=11JVRBuTryVbWExm4OBLLYzbjX5vVnUa1",
    "retinanet_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1yp1RrjbUQRnE5aDMrZ_5ico1pkLcxQ8U",
    "ssd300_vgg16": "https://drive.google.com/u/2/uc?id=1uwdZ2MiR3pUTjepmlC5ShrkUv_Ofji48",
    "ssdlite320_mobilenet_v3_large": "https://drive.google.com/u/2/uc?id=1kqVKds7Mm86XzLz48qKdO52ksmYSMizR",
}


def getImageClassModelList():
    return imageClassModelName


def getObjectDetectModelList():
    return objectDetectModelName
