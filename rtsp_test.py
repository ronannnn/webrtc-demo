import cv2

cap = cv2.VideoCapture("rtsp://192.168.1.103:8554/file")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
