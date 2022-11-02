# NGC image
FROM nvcr.io/nvidia/pytorch:22.02-py3

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 vim -y
RUN pip install aiohttp aiortc opencv-python==4.5.5.64
RUN pip install PyYAML tqdm matplotlib
RUN pip install pandas requests pillow
RUN pip install seaborn psutil ipython
RUN pip install scipy

COPY . .
CMD [ "python", "server.py"]
