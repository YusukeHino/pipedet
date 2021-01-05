FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \ 
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN echo 'fs.inotify.max_user_watches=131072' >> /etc/sysctl.conf
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN apt-get update -y
RUN apt-get install -y cmake
RUN apt-get install -y gcc g++
RUN apt-get install -y ffmpeg x264 libx264-dev
USER appuser
WORKDIR /home/appuser


ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN pip install --upgrade pip
RUN pip install --user numpy matplotlib pillow pandas requests scipy
# RUN pip install --user opencv-contrib-python-headless
RUN pip install --user hydra-core --upgrade
RUN pip install --user omegaconf
RUN pip install --user mypy

RUN python -m pip install --user grpcio

RUN python -m pip install --user grpcio-tools

# RUN git clone https://github.com/opencv/opencv.git -b 4.4.0

# RUN git clone https://github.com/opencv/opencv_contrib.git -b 4.4.0

# RUN mkdir opencv_build && \
# 	cd opencv_build && \
# 	cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv && \
# 	make -j5

# RUN cp /home/appuser/opencv_build/lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so /home/appuser/.local/lib/python3.7/site-packages

CMD ["/bin/bash"]

# Followings are from https://symfoware.blog.fc2.com/blog-entry-2345.html
RUN sudo apt autoremove libopencv-dev python-opencv

RUN wget --no-check-certificate https://raw.githubusercontent.com/milq/milq/master/scripts/bash/install-opencv.sh

#invalid
RUN sed -i -e "s/OPENCV_VERSION='4.2.0'/OPENCV_VERSION='4.1.0'/g" ./install-opencv.sh

RUN chmod +x install-opencv.sh

RUN ./install-opencv.sh