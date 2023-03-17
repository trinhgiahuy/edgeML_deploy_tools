from convert_constant import *

# NOTE: For imageClassificationModel, if have mean/std, function ImgClassPreProcess
# scaled to [0,1] before normalized
imgClassModelsCfg = {
    # https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
    "alexnet": {
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    "googlenet": {
        "preprocessing": {
            "resize_size": 256,
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",    
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    "efficientnet_b0":{
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},    
    },

    "densenet121":{
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES}, 
    },

    "squeezenet1_0":{
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},         
    },

    "inception_v3":{
        "preprocessing": {
            "resize_size": 342,  # It will compute height/width based on resize size
            "crop_size": 299,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},         
    },
    "shufflenet_v2_x0_5":{
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},         
    },

    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    # ? Recheck acc of verify img
    "mobilenet_v2": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    "mobilenet_v3_small": {
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "mobilenet_v3_large": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    "mnasnet0_5": {
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    "mnasnet0_75": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    "mnasnet1_0": {
         "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},       
    },
    "mnasnet1_3": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
    "resnet18": {
        "preprocessing": {
            "resize_size": 256,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "resnet50": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "resnet101": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "resnet152": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "resnext50_32x4d": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnext101_32x8d.html
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "resnext101_32x8d": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnext101_64x4d.html
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "resnext101_64x4d": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "wide_resnet50_2": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
    # https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet101_2.html
    # Resize size 232 , DEFAULT: IMAGENET1K_V2
    "wide_resnet101_2": {
        "preprocessing": {
            "resize_size": 232,  # It will compute height/width based on resize size
            "crop_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
}

objDetModelsCfg = {
    "fasterrcnn_resnet50_fpn": {
        "preprocessing": {
            "rescaled_01": True,
            "box_score_thresh": 0.5,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "label_last": False,
        },
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "preprocessing": {
            "rescaled_01": True,
            "box_score_thresh": 0.5,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "label_last": False,
        },
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "preprocessing": {
            "rescaled_01": True,
            "box_score_thresh": 0.5,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "label_last": False,
        },
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "preprocessing": {
            "rescaled_01": True,
            "box_score_thresh": 0.5,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "label_last": False,
        },
    },
    "fcos_resnet50_fpn": {
        "preprocessing": {
            "rescaled_01": True,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "score_thresh": 0.5,
            "label_last": True,
        },
    },
    "retinanet_resnet50_fpn": {
        "preprocessing": {
            "rescaled_01": True,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "score_thresh": 0.5,
            "label_last": True,
        },
    },
    "retinanet_resnet50_fpn_v2": {
        "preprocessing": {
            "rescaled_01": True,
            # Test min_size/max_size from torchvision source code
            "min_size": 800,
            "max_size": 1333,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "score_thresh": 0.5,
            "label_last": True,
        },
    },
    "ssd300_vgg16": {
        "preprocessing": {
            "rescaled_01": True,
            # By default, the model will resize input into (300,300) before feed into backbone
            "min_size": 300,
            "max_size": 300,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "score_thresh": 0.5,
            "label_last": True,
        },
    },
    "ssdlite320_mobilenet_v3_large": {
        "preprocessing": {
            "rescaled_01": True,
            # By default, the model will resize input into 320x320 before feed into MobileNetV3 Large backbone
            "min_size": 320,
            "max_size": 320,
        },
        "postprocessing": {
            "class_names": COCO_INSTANCE_CATEGORY_NAMES,
            "score_thresh": 0.5,
            "label_last": True,
        },
    },
    ## Custom models from github
    "tinyYOLOv2": {
        "preprocessing": {"custom": True, "new_height": 416, "new_width": 416},
        "postprocessing": {
            "score_thresh": 0.5,
            "anchors": [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
            "class_names": TINY_YOLO_LABEL_CLASSES,
        },
    },
    
    "tinyYOLOv3": {
        "preprocessing": {
            "custom": True, 
            "new_height": 416, 
            "new_width": 416,
            "keep_ratio": True,
            "pad_constant": 128,
            "rescaled_01": True,
            "interpolation": "cubic"
        },
        "postprocessing": {
            "score_thresh": 0.5,
            "class_names": COCO_CATEGORY_NAMES,
        },
    },
}

humanPoseModelsCfg = {
    "lightweightHumanPose":{
        "preprocessing": {
            "custom": True, 
            "resize_size": 256,
            "mean":[128,128,128],
            "scale":1/256,                   # In preprocess function, will take np.float32(1/256)
            "stride":8,
            
            #"cpu": True ???
        },
        "postprocessing": {
            "unsample_ratio":4,
        }
    }
}

semanSegmentModelsCfg = {
    "lraspp_mobilenet_v3_large": {
        "preprocessing": {
            "resize_size": 520,  # It will compute height/width based on resize size
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear",
        },
        "postprocessing": {"class_names": IMAGENET_CLASSES},
    },
}

customOnnxModel = {
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov2
    "tinyYOLOv2" : "/home/user/Downloads/tinyYOLOv2.onnx",

    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3
    "tinyYOLOv3" : "/home/user/Downloads/tinyYOLOv3.onnx",    

    #https://github.com/TheYoungBeast/YoloV4-Tiny-ONNX
    # "tinyYOLOv4" : "/home/user/Downloads/yolov4-tiny-single-batch.onnx"
    "tinyYOLOv4" : "/home/user/github_utils/darknet2onnx/yolov4-tiny.onnx",

    "YOLOv4":"/home/user/Downloads/yolov4.onnx",

    "yolox_nano":"/home/user/Yolox/yolox_nano.onnx",

    "yolox_tiny": "/home/user/Yolox/yolox_tiny.onnx",

    "yolox_s": "/home/user/Yolox/yolox_s.onnx",

    # "yolox_x": "/home/user/Yolox/yolox_x.onnx"

    "lightweightHumanPose": "/home/user/pose_est/lightweight-human-pose-estimation.pytorch/lightweightHumanPose.onnx"
}

pytorchOnnxModel_38 = {
    "yolov5n": "/home/user/YOLOv5/yolov5/optset13/yolov5n.onnx",
    "yolov5n6": "/home/user/YOLOv5/yolov5/optset13/yolov5n6.onnx",
    "yolov5s": "/home/user/YOLOv5/yolov5/optset13/yolov5s.onnx",
    "yolov5s6": "/home/user/YOLOv5/yolov5/optset13/yolov5s6.onnx",
}

pytorchOnnxModel_39 = {
    "yolov5n": "/home/user/Yolov5-ONNX/yolov5n.onnx",
    "yolov5n6": "/home/user/Yolov5-ONNX/yolov5n6.onnx",
    "yolov5s" : "/home/user/Yolov5-ONNX/yolov5s.onnx",
    "yolov5s6" : "/home/user/Yolov5-ONNX/yolov5s6.onnx",
}   
