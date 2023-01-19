# first, import all necessary modules
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import torch
import torchvision
import time
from time import monotonic
import click
import cv2


def setup_pipeline(nn_model_path):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    # Next, we want a neural network that will produce the detections
    detection_nn = pipeline.createNeuralNetwork()
    # Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
    detection_nn.setBlobPath(nn_model_path)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    # Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.preview.link(detection_nn.input)

    # Create outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("nn_input")
    xout_rgb.input.setBlocking(False)

    detection_nn.passthrough.link(xout_rgb.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    xout_nn.input.setBlocking(False)

    detection_nn.out.link(xout_nn.input)

    return pipeline


def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nxk (x1, y1, x2, y2, conf_cls_1, conf_cls_2, etc.)
    """
    prediction=torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    boxes = prediction[:, :4]
    scores = prediction[:, 4:]
    nc = prediction.shape[1] - 4  # number of classes
    keep = []
    for c in range(nc):
        valid_indices = torch.nonzero(torch.where(scores[:, c] > conf_thres, 1, 0), as_tuple=True)[0]
        if valid_indices.shape[0] > 0:
            nms = torchvision.ops.boxes.nms(boxes[valid_indices], scores[valid_indices, c], iou_thres)
            keep.append(prediction[valid_indices][nms])
    if keep:
        keep = torch.cat(keep, dim=0)
        return keep[torch.argsort(torch.max(keep[:, 4:], dim=1).values, descending=True)].cpu().numpy()
    else:
        return np.array([])

@click.command()
@click.option("-nn", "--nn-model", type=click.Path(exists=True),
              help="Model path for inference", default='models/mb2-ssd-lite-2021-4.blob')
@click.option("--threshold", type=click.FLOAT,
              help="Detection threshold", default=0.6)
@click.option("--max-boxes", type=click.INT,
              help="Maximum number of boxes to draw", default=1)
def main(nn_model, threshold, max_boxes):

    frame = None
    bird_boxes = []

    pipeline = setup_pipeline(nn_model)

    # Main host-side application loop
    with dai.Device() as device:
        # Start pipeline
        device.startPipeline(pipeline)

        q_rgb = device.getOutputQueue("nn_input", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.tryGet()
            out_nn = q_nn.tryGet()

            if in_rgb is not None:
                # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                # Also, the array is transformed from CHW form into HWC
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)

            if out_nn is not None:
                # when data from nn is received, it is also represented as a 1D array initially, just like rgb frame
                bboxes = np.array(out_nn.getLayerFp16('boxes')).reshape(3000, 4)
                scores = np.array(out_nn.getLayerFp16('scores')).reshape(3000, 2)
                bird_boxes = non_max_suppression(np.concatenate((bboxes, scores[:, 1:]), axis=1), conf_thres=threshold)[:max_boxes]

            if frame is not None:
                for raw_bbox in bird_boxes:
                    # for each bounding box, we first normalize it to match the frame size
                    bbox = frame_norm(frame, raw_bbox[:4])
                    # and then draw a rectangle on the frame to show the actual result
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # After all the drawing is finished, we show the frame on the screen
                cv2.imshow("preview", frame)

            # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
