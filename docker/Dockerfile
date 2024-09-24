FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean -y

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model.pth /opt/algorithm/

ENTRYPOINT python -m process $0 $@
