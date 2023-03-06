COCODriveLink = "https://drive.google.com/u/2/uc?id=1gE5C1UXcCyogZU3K95tn-bXfuF8a-8h4"

imageClassModelName = [
    "alexnet",
    "googlenet",
    "efficientnet_b0",
    "densenet121",
    "squeezenet1_0",
    "inception_v3",
    "maxvit_t",
    "shufflenet_v2_x0_5",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_3",
    "resnet18",
    # # "resnet50",
    # # "resnet101",
    # # "resnet152",
    # "resnext50_32x4d",
    # # "resnext101_32x8d",
    # # "resnext101_64x4d",
    # "wide_resnet50_2",
    # # "googlenet",
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

customobjectDetectModelName = ["tinyYOLOv2"]

modelLinkDrive = {
    "alexnet": "https://drive.google.com/u/2/uc?id=1FPfEM7T1lXfezz_9jWY0jnp4KBq16wrz",
    "googlenet":"https://drive.google.com/u/2/uc?id=1s9BH24hhzZBwWlzqZNicQuRrqRlIqO8f",
    "efficientnet_b0":"https://drive.google.com/u/2/uc?id=19sv8_C36BN1lK_dZ-ARukN0pvK4pUmyS",
    "densenet121":"https://drive.google.com/u/2/uc?id=1234hYCBK0VgZOLIZIqM4zMGNIphrdFKx",
    "squeezenet1_0":"https://drive.google.com/u/2/uc?id=1Nb3I2lsJ1VzrmYPjjwrWkYQ3RAdNCCoC",
    "inception_v3":"https://drive.google.com/u/2/uc?id=1YUqQ1dOKSpBWX8TRUZ7LgNiZCWCOEOKd",
    "maxvit_t":"https://drive.google.com/u/2/uc?id=1kKfyEJLyd30vLARuf16SxkIu3WvwA3rb",
    "shufflenet_v2_x0_5":"https://drive.google.com/u/2/uc?id=1_GEQRc8Oq_Ikg2uSIykgqxPonGuMwdBc",
    "mobilenet_v2":"https://drive.google.com/u/2/uc?id=1TuuDSv20vhxCJrI5ZNkJjYrKIm2WGa6K",
    "mobilenet_v3_small":"https://drive.google.com/u/2/uc?id=1D0dKvbrF4SIcJ6q-KEauWDa3cO2ydBnH",
    "mobilenet_v3_large":"https://drive.google.com/u/2/uc?id=13FrrsBTH9wvLFWPBPEgkkAinUg35sZDJ",
    "mnasnet0_5":"https://drive.google.com/u/2/uc?id=1yhxMtKafHwTC23cwyqIbGnCNzwNzXLGs",
    "mnasnet0_75":"https://drive.google.com/u/2/uc?id=1FVZylLcnsU4NWCujk2WiGZ0geh18JH9m",
    "mnasnet1_3":"https://drive.google.com/u/2/uc?id=1TJzku6MXIU1jRhHfB_TlCdgi26iQZunt",
    "resnet18":"https://drive.google.com/u/2/uc?id=17m7CuXcnlHkco5CXrNhvOwLVbfpiD4Ix",

    "resnext50_32x4d":"https://drive.google.com/u/2/uc?id=1NP13J-kbD5Hb7zs4lSyWNGb4S5teu9t8",
    "wide_resnet50_2":"https://drive.google.com/u/2/uc?id=1JA4OQHJfMPtgECIpaZ71pXROO09Qpp3f",

    # "fasterrcnn_resnet50_fpn_v2": "https://drive.google.com/u/2/uc?id=1DAqbZmcp8a8WJbWhPZmBBhHFkBGtsVMh",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "https://drive.google.com/u/2/uc?id=18m3-apGnDGwE_X5_AuojFhiYbeJU8mlq",
    "fasterrcnn_mobilenet_v3_large_fpn": "https://drive.google.com/u/2/uc?id=1vtdn-fpadNaBBscOxa1vCrhAhX1ADwEK",
    # "fasterrcnn_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1gSyeMNMyXSooTnTFR8meNhR3N-F1HUkb",
    # "fcos_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1IzsgAfMs4uxib42BFv9S8ybWP2jcTdMr",
    # "retinanet_resnet50_fpn_v2": "https://drive.google.com/u/2/uc?id=1Zv3wTj8EbVMUGqRpjfNPOmRNAPXaIhdv",
    # "retinanet_resnet50_fpn": "https://drive.google.com/u/2/uc?id=1NEDQCGFQO5z2CmNbyNBx_wVSBjSM1LuO",
    "ssd300_vgg16": "https://drive.google.com/u/2/uc?id=1eoC_3tH_s0HbuMTJ-j2zEBr1uTTFe529",
    "ssdlite320_mobilenet_v3_large": "https://drive.google.com/u/2/uc?id=14qxtYiZ6euKSYRgePFK4XI9UQAaopvCO",
    "tinyYOLOv2": "https://drive.google.com/u/2/uc?id=1viejka4pVIKRsHCsd3ljvkParN9DxQ8r",
}

JetsonNanoLinkDrive = {
    # "fasterrcnn_resnet50_fpn":"https://drive.google.com/u/2/uc?id=13_HghVj1xNuu577A3K9c3oekeC_zIZ_a",
    "fasterrcnn_mobilenet_v3_large_fpn": "https://drive.google.com/u/2/uc?id=1_rQk4N9OezQBbuaPFnqm1NCEVdWFuqwg",
    "ssd300_vgg16": "https://drive.google.com/u/2/uc?id=1J-CSHqZy8-WM6r2JKk3-bO3Lc4N6kXoG",
    "ssdlite320_mobilenet_v3_large": "https://drive.google.com/u/2/uc?id=1BCneFf2fOYJumSjq4IS5iV4vN6w7kaV7",
}

JetsonTX2LinkDrive={
    "alexnet":"https://drive.google.com/u/2/uc?id=1EQ4EUmzO_z-btbYL8WoanHTjsdfoW-7N"
}


def getImageClassModelList():
    return imageClassModelName

def getObjectDetectModelList():
    return objectDetectModelName
    
def getObjectDetectCustomModelList():
    return customobjectDetectModelName