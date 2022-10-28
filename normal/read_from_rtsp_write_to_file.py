import time

import cv2
from yolo_object_detector import YoloObjectDetector


def read_from_rtsp_write_to_file():
    cap = cv2.VideoCapture("rtsp://192.168.1.100:8554/file")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    yod = YoloObjectDetector(True)
    prev_frame_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = yod.plot_boxes(frame)
        # draw fps
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        frame = cv2.putText(frame, "FPS: {}".format(str(int(fps))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5, cv2.LINE_AA)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_from_rtsp_write_to_file()
