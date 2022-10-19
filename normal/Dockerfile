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
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple aiohttp aiortc opencv-python
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple PyYAML tqdm matplotlib
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas requests pillow
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple seaborn psutil

# install customized python lib
WORKDIR /app
COPY . .
RUN pip3 install torch-1.14.0a0+git1707438-cp39-cp39-linux_aarch64.whl

# run
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
CMD [ "python3.9", "server.py"]
