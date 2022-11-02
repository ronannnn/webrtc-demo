# Webrtc demo with yolo object detection

## Reference
[python webrtc](https://github.com/aiortc/aiortc/tree/main/examples/server)

## Debug locally
```bash
RTSP_SERVER_IP=192.168.1.103 RTSP_SERVER_PORT=8554 RTSP_SERVER_PATH=file ENABLE_YOLO=true ENABLE_GPU=false YOLO_MODEL_NAME=yolov5n.pt python3 server.py
```
Or you can `export` some environment variables and
```bash
python3 server.py
```