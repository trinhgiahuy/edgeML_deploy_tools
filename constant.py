COCODriveLink = "https://drive.google.com/u/2/uc?id=1gE5C1UXcCyogZU3K95tn-bXfuF8a-8h4"

imageClassModelName = [
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
    "mnasnet1_0",
    "mnasnet1_3",
    "resnet18",
]

objectDetectModelName = [
    # "fasterrcnn_mobilenet_v3_large_320_fpn",
    # "fasterrcnn_mobilenet_v3_large_fpn",
    # "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
]

customobjectDetectModelName = [
    "tinyYOLOv2", 
    "tinyYOLOv3", 
    'yolov5n',
    'yolov5n6',
    'yolov5s',
    'yolov5s6',
]

YOLOXObjectDetectOnnxModelName = [
    'yolox_nano',
    'yolox_tiny',
    'yolox_s'
]

HumanPoseOnnxModelName = [
    'lightweightHumanPose'
]

# For Pi3/4
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
    "mnasnet1_0":"https://drive.google.com/u/2/uc?id=1mtRGQhOS7S_AEBMtLuP6dbugdcvjGz4C",
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
    "tinyYOLOv3": "https://drive.google.com/u/2/uc?id=1-_c1V8s_vOjLO2D5ypg1W6kgiFPbvmwm",

    "yolov5n": "https://drive.google.com/u/2/uc?id=1U8yozXeRBwaAoSwQPcBHguvqduBtNfJR",
    "yolov5s": "https://drive.google.com/u/2/uc?id=1CtLFTjVQeYso3tb074PZEmo-NouZQWWQ",
    "yolov5m": "https://drive.google.com/u/2/uc?id=1fir_93j-9kqeAxT6ZMN6qcGwmP-LUkmB",
    "yolov5n6": "https://drive.google.com/u/2/uc?id=1GQHHd7_QKjWhU4H4Sk8qDN5xAnsOqqaP",
    "yolov5s6":"https://drive.google.com/u/2/uc?id=1_OtkKu1FWgLU5IicPE__3aok0pzueYno",
    "yolov5m6":"https://drive.google.com/u/2/uc?id=16e9tS1a_wPFXmn8YLMc0MVnm034Q_anI",

    "yolox_nano": "https://drive.google.com/u/2/uc?id=1a8oy7feGtl3G1FhBO3c82ZAkIUaQ5uKx",
    "yolox_tiny": "https://drive.google.com/u/2/uc?id=1SLX-36DVvwPxAG7HkKWdZRoMU2hWpYJ5",
    "yolox_s" : "https://drive.google.com/u/2/uc?id=1UjYOqulLo8uMih7ViZhaMTfKBcT5w60K",

    "lightweightHumanPose": "https://drive.google.com/u/2/uc?id=1G-RCf8A-tmM0yOMHG54cLP15g2Yvf6Dn"
}

JetsonNanoLinkDrive = {
    # "fasterrcnn_resnet50_fpn":"https://drive.google.com/u/2/uc?id=13_HghVj1xNuu577A3K9c3oekeC_zIZ_a",
    "fasterrcnn_mobilenet_v3_large_fpn": "https://drive.google.com/u/2/uc?id=1_rQk4N9OezQBbuaPFnqm1NCEVdWFuqwg",
    "ssd300_vgg16": "https://drive.google.com/u/2/uc?id=1J-CSHqZy8-WM6r2JKk3-bO3Lc4N6kXoG",
    "ssdlite320_mobilenet_v3_large": "https://drive.google.com/u/2/uc?id=1BCneFf2fOYJumSjq4IS5iV4vN6w7kaV7",
}

JetsonTX2LinkDrive={
    "alexnet":"https://drive.google.com/u/2/uc?id=1EQ4EUmzO_z-btbYL8WoanHTjsdfoW-7N",
    "googlenet":"https://drive.google.com/u/2/uc?id=1siJe1hsl5tb3DlILG0ayBLirmFfM0EZc",
    "efficientnet_b0":"https://drive.google.com/uc?id=1TscfrRJJ1UfNeQNTVB8PuDc9n1VWEtkk",
    "densenet121":"https://drive.google.com/uc?id=1sMsanepR7moj1rLhkEA0i37rqWilP4L9",
    "squeezenet1_0":"https://drive.google.com/uc?id=1bYaS0Cg34f2P6wumh4ehQNwS7wNsCMks",
    "inception_v3":"https://drive.google.com/uc?id=1pPR0w6c6LjIG16BRfsVysmfwB34pGR0W",
    "shufflenet_v2_x0_5":"https://drive.google.com/uc?id=1zyyaYjA_J-eMrTYMIxoN1pGFv1EyY-3Y",
    "mobilenet_v2":"https://drive.google.com/uc?id=1Drw0VTONIe9oHsaHA7lli6eYnNz3MI3d",
    "mobilenet_v3_small":"https://drive.google.com/uc?id=1KJn9D68VMorkJUyvLBloGf0lXpMAsN5Y",
    "mobilenet_v3_large":"https://drive.google.com/uc?id=1QE34fl5zkqrm9UGygNy__bkzEBeI8-fp",
    "mnasnet0_5":"https://drive.google.com/u/2/uc?id=1tvm21tMadIyJI4L_nsota5GjXP1shArC",
    "mnasnet0_75":"https://drive.google.com/uc?id=1uu5pOM824FFY8R0qh1vrw1TlMZTQN_k-",
    "mnasnet1_0":"https://drive.google.com/uc?id=1WR1rddRg3LcFbycD3kprXEYiTSl-gQ6H",
    "mnasnet1_3":"https://drive.google.com/uc?id=1hi_ixPsa7vXhbQE_vHfU8kyTrPxUUC_q",
    "resnet18":"https://drive.google.com/u/2/uc?id=1llezqtYeDA7kVew3hJyHY8RVhdBKIuks",

    "fasterrcnn_mobilenet_v3_large_fpn":"",
    "fasterrcnn_mobilenet_v3_large_320_fpn":"",
    "ssd300_vgg16":"",
    "ssdlite320_mobilenet_v3_large":"https://drive.google.com/u/2/uc?id=1lOweARggMEJyyLvyPzarLerX6ot3B1_c",

    "tinyYOLOv2": "https://drive.google.com/u/2/uc?id=1cdkVpZl8K56gIz12Y15_icPQ0hufd3_Y",
    "tinyYOLOv3":"https://drive.google.com/u/2/uc?id=1NsAujinS-It1plRg5OtOcHQBFfXrLWoW",
    
    "yolov5n":"https://drive.google.com/u/2/uc?id=1xzMtHQsJFK3SGQh-CRihbm2sjjs-7Z49",
    "yolov5n6":"https://drive.google.com/u/2/uc?id=1i2Ws6YPp85BgTCjSu36dC0PS7LaW9weQ",
    "yolov5s":"https://drive.google.com/u/2/uc?id=1o7eL8T2PtM5S5MOKrR0gW-dYunnk2Tcl",
    "yolov5s6":"https://drive.google.com/u/2/uc?id=1VsAPjTgQbdgZVZLFaLjwvc_9bR7o69J6",

    "yolox_nano":"https://drive.google.com/u/2/uc?id=1DAw6lxF6aUkw06Cu5yHH9A5Uq5dlsrxh",
    "yolox_tiny":"https://drive.google.com/u/2/uc?id=1aAE6l2BfGpgYSgwekmtX7Vpei9YtY1fy",
    "yolox_s":"https://drive.google.com/u/2/uc?id=1feOP-Y2JGNZt6Rd72fsS6cNMPffOQheE"
}

jetsonTensorLink ={
    "alexnet": "",
    "googlenet": "",
    "efficientnet_b0": "",
    "densenet121" : "",
    "squeezenet1_0" : "",
    "inception_v3" : "",
    "shufflenet_v2_x0_5" : "",
    "mobilenet_v2" : "",
    "mobilenet_v3_small" : "",
    "mobilenet_v3_large" : "",
    "mnasnet0_5" : "",
    "mnasnet0_75" : "",
    "mnasnet1_0" : "",
    "mnasnet1_3" : "",
    "resnet18" : "",
}

execProvider = [
    ""
]
def getImageClassModelList():
    return imageClassModelName

def getObjectDetectModelList():
    return objectDetectModelName
    
def getObjectDetectCustomModelList():
    return customobjectDetectModelName

def getYOLOXObjectDetectModelList():
    return YOLOXObjectDetectOnnxModelName

def getHumanPoseModelList():
    return HumanPoseOnnxModelName