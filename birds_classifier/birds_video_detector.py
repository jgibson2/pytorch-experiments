# first, import all necessary modules
from pathlib import Path
from time import monotonic

import click
import cv2
import depthai as dai
import numpy as np
from birds_detector import frame_norm, to_planar, non_max_suppression
import sys

def setup_pipeline(nn_model_path):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    # Create xLink input to which host will send frames from the video file
    xinFrame = pipeline.createXLinkIn()
    xinFrame.setStreamName("inFrame")

    # Next, we want a neural network that will produce the detections
    detection_nn = pipeline.createNeuralNetwork()
    # Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
    detection_nn.setBlobPath(nn_model_path)
    # Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
    xinFrame.out.link(detection_nn.input)

    # The same XLinkOut mechanism will be used to receive nn results
    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)

    return pipeline


@click.command()
@click.option("-v", "--video", type=click.Path(exists=True), help="Video file to run inference on")
@click.option("-nn", "--nn-model", type=click.Path(exists=True),
              help="Model path for inference", default='models/mb2-ssd-lite-2021-4.blob')
@click.option("--threshold", type=click.FLOAT,
              help="Detection threshold", default=0.6)
@click.option("--max-boxes", type=click.INT,
              help="Maximum number of boxes to draw", default=1)
def main(video, nn_model, threshold, max_boxes):

    frame = None
    bird_boxes = []

    pipeline = setup_pipeline(nn_model)

    # Main host-side application loop
    with dai.Device() as device:
        # Start pipeline
        device.startPipeline(pipeline)

        q_in = device.getInputQueue("inFrame")
        # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        cap = cv2.VideoCapture(str(video))
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
                # thresh = threshold
                # bird_box_indices = np.where(np.logical_and(scores[:, 1] > thresh, scores[:, 1] > scores[:, 0]))[0]
                # if bird_box_indices.shape[0] > 0:
                #     bird_box_indices = bird_box_indices[np.argsort(scores[bird_box_indices, 1][-max_boxes:])]
                #     bird_boxes = bboxes[bird_box_indices, :]
                # else:
                #     bird_boxes = []
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
