ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:6.2-triton
FROM ${BASE_IMAGE}

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y \
   python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
   python3 python3-pip python3.8-dev \
   python3-setuptools \
   cmake g++ build-essential libglib2.0-dev \
   libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev \
   python3-opencv libopencv-dev \
   libgstrtspserver-1.0-0 \
   gstreamer1.0-rtsp \
   gstreamer1.0-libav \
   gobject-introspection \
   gir1.2-gst-rtsp-server-1.0 \
   && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh

RUN cd /opt/nvidia/deepstream/deepstream/sources/ && git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

RUN cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/ && git submodule update --init

RUN apt-get install -y apt-transport-https ca-certificates -y && update-ca-certificates

RUN cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/3rdparty/gst-python/ && ./autogen.sh && make && make install

RUN cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings && mkdir build && cd build && cmake .. && make && pip3 install ./pyds-1.1.6-py3-none*.whl


