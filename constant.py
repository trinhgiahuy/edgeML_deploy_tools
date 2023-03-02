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
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    # "fasterrcnn_resnet50_fpn_v2", 
    # "fasterrcnn_resnet50_fpn",
    # "fcos_resnet50_fpn",
    # "retinanet_resnet50_fpn_v2",
    # "retinanet_resnet50_fpn",
]

modelLinkDrive = {
    "alexnet": "https://drive.google.com/u/2/uc?id=1tv9NPj9RaYAsek8hCnsFx_YPdfyt9c6t",
    # "fasterrcnn_resnet50_fpn_v2": "https://drive.google.com/u/2/uc?id=1DAqbZmcp8a8WJbWhPZmBBhHFkBGtsVMh",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "https://drive.google.com/u/2/uc?id=18m3-apGnDGwE_X5_AuojFhiYbeJU8mlq",
    "fasterrcnn_mobilenet_v3_large_fpn": "https://drive.google.com/u/2/uc?id=1vtdn-fpadNaBBscOxa1vCrhAhX1ADwEK",
    # "fasterrcnn_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1gSyeMNMyXSooTnTFR8meNhR3N-F1HUkb",
    # "fcos_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1IzsgAfMs4uxib42BFv9S8ybWP2jcTdMr",
    # "retinanet_resnet50_fpn_v2": "https://drive.google.com/u/2/uc?id=1Zv3wTj8EbVMUGqRpjfNPOmRNAPXaIhdv",
    # "retinanet_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1NEDQCGFQO5z2CmNbyNBx_wVSBjSM1LuO",
    "ssd300_vgg16": "https://drive.google.com/u/2/uc?id=1eoC_3tH_s0HbuMTJ-j2zEBr1uTTFe529",
    "ssdlite320_mobilenet_v3_large": "https://drive.google.com/u/2/uc?id=14qxtYiZ6euKSYRgePFK4XI9UQAaopvCO",
}

JetsonNanoLinkDrive = {
    "fasterrcnn_resnet50_fpn":"https://drive.google.com/u/2/uc?id=13_HghVj1xNuu577A3K9c3oekeC_zIZ_a"
}

def getImageClassModelList():
    return imageClassModelName


def getObjectDetectModelList():
    return objectDetectModelName
