# docker build --network=host -t sekidocker.rnd.ki.sw.ericsson.se/proj-tnoam-test/hfmpred:<username>-<date> . -f ./docker/Dockerfile

FROM armdockerhub.rnd.ericsson.se/ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive

ARG ARM_3PPS="https://arm.seli.gic.ericsson.se/artifactory/proj-ccs-ci-delivery-generic-local/3pps"

LABEL org.opencontainers.image.title="HFMpred_app-egumsan" \
  org.opencontainers.image.description="An app for 1D heat flux prediction" \
  org.opencontainers.image.created="2022-06-08" \
  org.opencontainers.image.authors="Sanjin Gumbarevic"

RUN apt-get -y update && apt-get -y install apt-utils

RUN apt-get install -y wget curl \
  python3-dev python3-venv python3-pip python3-setuptools python3-wheel python3-requests python-is-python3 &&\
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /etc/requirements.txt

RUN python3 -m pip install -r /etc/requirements.txt

ENV BUILD_USER hfmpred

ARG host_uid=7286721
ARG host_gid=64000
ARG rt_tn_gid=22070
RUN echo "%eusers   ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
  groupadd -g $host_gid eusers && \
  groupadd -g $rt_tn_gid rt_tn && \
  useradd  -m -g $host_gid -g $rt_tn_gid -s /bin/bash -u $host_uid hfmpred && \
  mkdir -p /home/hfmpred/app && \
  chown -R hfmpred:users /home/hfmpred && \
  truncate -s 0 /var/log/faillog /var/log/lastlog

USER hfmpred

COPY . /hfmpred/app
WORKDIR /hfmpred/app

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]
CMD ["HFMpred_app.py"]
