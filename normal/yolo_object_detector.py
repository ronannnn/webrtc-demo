import random
import time
import logging as log

import torch
import cv2


class YoloObjectDetector:
    """
    Class implements Yolo5 model to make inferences on a YouTube video using Opencv2.
    """

    def __init__(self):
        # https://docs.ultralytics.com/tutorials/pytorch-hub/
        torch.set_num_threads(32)
        self.model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
        self.classes = self.model.names
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.color_map = {}
        log.info("Using Device: %s", self.device)

    def plot_boxes(self, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.

        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels plotted on it.
        """

        # Get labels and coordinates of predicted objects
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        # draw boxes
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(len(labels)):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = self.get_box_color(self.classes[int(labels[i])])
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def get_box_color(self, label):
        if label in self.color_map:
            return self.color_map[label]
        else:
            self.color_map[label] = random.choices(range(256), k=3)
            return self.color_map[label]

    def live_demo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        prev_frame_time = time.time()
        while True:
            success, frame = cap.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.flip(frame, 1)

            # get fps
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # detect lane line
            frame = self.plot_boxes(frame)
            frame = cv2.putText(frame, "FPS: {}".format(str(int(fps))), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    YoloObjectDetector().live_demo()
