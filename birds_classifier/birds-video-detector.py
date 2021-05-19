# first, import all necessary modules
from pathlib import Path
from time import monotonic

import cv2
import depthai as dai
import numpy as np
import sys

INITIAL_THRESHOLD = 0.80
TRACKING_THRESHOLD = 0.70
MAX_BOXES = 1
HYSTERESIS = 3

videoPath = str(Path(sys.argv[1]).resolve().absolute())

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = dai.Pipeline()

# Create xLink input to which host will send frames from the video file
xinFrame = pipeline.createXLinkIn()
xinFrame.setStreamName("inFrame")

# Next, we want a neural network that will produce the detections
detection_nn = pipeline.createNeuralNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
detection_nn.setBlobPath(
    str((Path(__file__).parent / Path('pytorch-ssd/models/mb2-ssd-lite.blob')).resolve().absolute()))
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
xinFrame.out.link(detection_nn.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Here, some of the default values are defined. Frame will be an image from "rgb" stream, bboxes will contain nn results
frame = None
bboxes = []
bird_boxes = []
counter = HYSTERESIS + 1


# Since the bboxes returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
# receive the actual position of the bounding box on the image
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


# Main host-side application loop
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    q_in = device.getInputQueue("inFrame")
    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        curr_shape = frame.shape[:2]
        side_len = min(*curr_shape)
        frame = frame[(curr_shape[0] - side_len) // 2: curr_shape[0] - (curr_shape[0] - side_len) // 2,
              (curr_shape[1] - side_len) // 2: curr_shape[1] - (curr_shape[1] - side_len) // 2]

        img = dai.ImgFrame()
        img.setData(to_planar(frame, (300, 300)))
        img.setTimestamp(monotonic())
        img.setWidth(300)
        img.setHeight(300)
        q_in.send(img)
        out_nn = q_nn.get()

        if out_nn is not None:
            # when data from nn is received, it is also represented as a 1D array initially, just like rgb frame
            bboxes = np.array(out_nn.getLayerFp16('boxes')).reshape(3000, 4)
            scores = np.array(out_nn.getLayerFp16('scores')).reshape(3000, 2)
            thresh = INITIAL_THRESHOLD if counter > HYSTERESIS else TRACKING_THRESHOLD
            bird_box_indices = np.where(np.logical_and(scores[:, 1] > thresh, scores[:, 1] > scores[:, 0]))[0]
            # bird_box_indices = np.where(scores[:, 1] > THRESHOLD)[0]
            if bird_box_indices.shape[0] > 0:
                bird_box_indices = bird_box_indices[np.argsort(scores[bird_box_indices, 1][-MAX_BOXES:])]
                bird_boxes = bboxes[bird_box_indices, :]
                counter = 0
                # print('Got birds!')
            else:
                counter += 1
                if counter > HYSTERESIS:
                    bird_boxes = []

        if frame is not None:
            for raw_bbox in bird_boxes:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frame_norm(frame, raw_bbox)
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break
