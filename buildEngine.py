import os
from loguru import logger
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import argparse

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)



def buildTensorRTEngine(modelOnnxPathName):
    
    if not os.path.exists(modelOnnxPathName):
        print(
                "ONNX file {} not found, please generate it.".format(modelOnnxPathName)
            )
        exit(0)
    model_name = modelOnnxPathName.split('/')[-1]
    logger.warning(model_name)

    engineFilePath = modelOnnxPathName.replace('.onnx','.trt')


    if os.path.exists(engineFilePath):
        logger.info(f"Engine file for model {model_name} exists!")

        return 1
    else:
        
        # Trt engine file does not exist. Build it
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            builder.max_batch_size = 1
            config.max_workspace_size = 1 << 28  # 256MiB


            print("Loading ONNX file from path {}...".format(modelOnnxPathName))
            with open(modelOnnxPathName, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
        
        
            logger.info(f'engineFilePath: {engineFilePath}')

            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(modelOnnxPathName))
            plan = builder.build_serialized_network(network, config)
            # engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engineFilePath, "wb") as f:
                f.write(plan)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numIteration", type=int, default=5000, help="Number of iterations to run"
    )
    parser.add_argument(
        "--modelFn", type=str, required=True, help="File name of the model archive"
    )
    parser.add_argument(
        "--application",
        type=str,
        required=True,
        help="ML application (image_class,object_detect)",
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Runtime prefix (onnx,trt)"
    )
    args = parser.parse_args()
   
    numIteration = args.numIteration
    # modelFn = "mnasnet0_5"
    # application = "image_class"
    # prefix = "onnx"
    modelFn = args.modelFn
    application = args.application
    prefix = args.prefix
    execProvider = "rt"

    cwd = os.getcwd()

    # Same directory with google drive
    predir = application
    # COCO_verify_dir = f"{cwd}/COCO_5000_imgs"

    # modelDir = cwd/image_class_trt/
    modelDir = f"{cwd}/{predir}_{prefix}"
    # output_dir = f"{cwd}/csv_output"

    # tensor_output_dir = f"{cwd}/tensorRT/rt"
    tesor_output_dir = f"{cwd}/tensorRT/{execProvider}"

    # ONNX FILE GENERATED BY PYTHON3.6 FOR ALL JETSONS
    # modelOnnxPathName = f"{cwd}/image_class_trt/alexnet.onnx"
    modelOnnxPathName = os.path.join(modelDir, modelFn + ".onnx")
    print(modelOnnxPathName)

    # Will check if trt engine file exists
    buildTensorRTEngine(modelOnnxPathName)
