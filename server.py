import argparse
import asyncio
import json
import logging as log
import os
import time

import cv2
from av import VideoFrame

from aiohttp import web

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay

from yolo_object_detector import YoloObjectDetector

ROOT = os.path.dirname(__file__)

relay = None
webcam = None

# ENV VARIABLES
# rtsp-related
rtsp_server_ip = os.getenv("RTSP_SERVER_IP")
rtsp_server_ip = rtsp_server_ip if rtsp_server_ip is not None else "10.70.185.63"
rtsp_server_port = os.getenv("RTSP_SERVER_PORT")
rtsp_server_port = rtsp_server_port if rtsp_server_port is not None else "8554"
rtsp_server_path = os.getenv("RTSP_SERVER_PATH")
rtsp_server_path = rtsp_server_path if rtsp_server_path is not None else "file"
rtsp_addr = "rtsp://{}:{}/{}".format(rtsp_server_ip, rtsp_server_port, rtsp_server_path)

# yolo-related
enable_yolo = os.getenv("ENABLE_YOLO")
enable_yolo = enable_yolo.lower() if enable_yolo is not None else 'false'
enable_yolo = True if enable_yolo == 'true' else False

enable_gpu = os.getenv("ENABLE_GPU")
enable_gpu = enable_gpu.lower() if enable_gpu is not None else 'false'
enable_gpu = True if enable_gpu == 'true' else False

yolo_model_name = os.getenv("YOLO_MODEL_NAME")
yolo_model_name = yolo_model_name.lower() if yolo_model_name is not None else 'yolov5n.pt'


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from another track.
    """

    kind = "video"

    def __init__(self, track, enable_object_detection, enable_gpu):
        super().__init__()  # don't forget this!
        self.track = track
        self.prev_frame_time = time.time()
        self.enable_object_detection = enable_object_detection
        if enable_object_detection:
            self.yod = YoloObjectDetector(yolo_model_name, enable_gpu)

    async def recv(self):
        frame = await self.track.recv()

        img = frame.to_ndarray(format="bgr24")

        # draw fps
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time

        # draw object box
        if self.enable_object_detection:
            img = self.yod.plot_boxes(img)
        img = cv2.putText(img, "FPS: {}".format(str(int(fps))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5, cv2.LINE_AA)

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


def create_local_tracks():
    global relay, webcam
    options = {"framerate": "30", "video_size": "640x480"}
    if relay is None:
        webcam = MediaPlayer(rtsp_addr, format="rtsp", options=options)
        relay = MediaRelay()
    return relay.subscribe(webcam.video)


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    pc.addTrack(VideoTransformTrack(create_local_tracks(), enable_object_detection=enable_yolo, enable_gpu=enable_gpu))

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )

    args = parser.parse_args()
    log.basicConfig(level=log.INFO)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port, ssl_context=None)
