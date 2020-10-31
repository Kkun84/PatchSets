# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Timezone setting
RUN apt-get update && apt-get install -y --no-install-recommends tzdata

# Install something
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-add-repository -y ppa:fish-shell/release-3
RUN apt-get update && apt-get install -y --no-install-recommends fish

RUN apt-get update && apt-get install -y --no-install-recommends nano git sudo curl

# OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev

# Install Python library
COPY requirements.txt /
RUN pip install -r /requirements.txt

ARG UID
ARG GID
ARG USER
ARG PASSWORD
RUN groupadd -g ${GID} ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd
USER ${USER}
