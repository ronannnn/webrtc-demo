FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9
RUN python3.9 --version
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt-get install -y python3.9-distutils
RUN python3.9 get-pip.py
# make sure pip3 install package for python3.9
RUN pip3 --version

RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN apt-get install -y libopencv-dev
RUN apt-get install -y ffmpeg
RUN apt-get install -y libatlas-base-dev

# install official python lib
RUN apt-get install -y libpython3.9-dev
RUN pip3 install aiohttp aiortc opencv-python
RUN pip3 install PyYAML tqdm matplotlib
RUN pip3 install pandas requests pillow
RUN pip3 install seaborn psutil ipython
RUN pip3 install scipy

# install customized python lib
WORKDIR /app
COPY torch-1.14.0a0+gitb5bdc34-cp39-cp39-linux_aarch64.whl ./
RUN pip3 install torch-1.14.0a0+gitb5bdc34-cp39-cp39-linux_aarch64.whl
COPY torchvision-0.15.0a0+7a62a54-cp39-cp39-linux_aarch64.whl ./
RUN pip3 install torchvision-0.15.0a0+7a62a54-cp39-cp39-linux_aarch64.whl

RUN apt-get install libcudnn8=8.5.0.96-1+cuda11.7
RUN apt-get install libcudnn8-dev=8.5.0.96-1+cuda11.7

# run
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/:/usr/local/cuda/lib64:/usr/local/cuda/include:/usr/lib/aarch64-linux-gnu"
COPY . .
CMD [ "python3.9", "server.py"]
