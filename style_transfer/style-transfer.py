# first, import all necessary modules
from pathlib import Path
import cv2
import depthai
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('input_model_blob', help='Path to model blob', type=str)
args = parser.parse_args()

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()
pipeline.setOpenVINOVersion(depthai.OpenVINO.VERSION_2021_1)

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(224, 224)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

# Next, we want a neural network that will produce the detections
nn = pipeline.createNeuralNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
nn.setBlobPath(str(Path(args.input_model_blob).resolve().absolute()))
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(nn.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
device = depthai.Device(pipeline)
# And start. From this point, the Device will be in "running" mode and will start sending data via XLink
device.startPipeline()

# To consume the device results, we get two output queues from the device, with stream names we assigned earlier
q_nn = device.getOutputQueue("nn")

# Here, some of the default values are defined. Frame will be an image from "rgb" stream, bboxes will contain nn results
style_frame = None
startTime = time.monotonic()
counter = 0
fps = 0

# Main host-side application loop
while True:
    # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
    in_nn = q_nn.get()

    counter += 1
    current_time = time.monotonic()
    if (current_time - startTime) > 1:
        fps = counter / (current_time - startTime)
        counter = 0
        startTime = current_time

    if in_nn is not None:
        style_frame = np.clip(np.array(in_nn.getFirstLayerFp16()), 0, 255)\
                          .reshape((3, 224, 224)).transpose(1, 2, 0) / 255.0

    if style_frame is not None:
        cv2.imshow("preview", cv2.putText(cv2.UMat(style_frame), "NN fps: {:.2f}".format(fps), (2, style_frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255)))

    # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
    if cv2.waitKey(1) == ord('q'):
        break