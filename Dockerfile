FROM --platform=linux/amd64 nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV RUNNING_USER=nginx
ENV THEAPP=/theapp
ENV HF_DATASETS_CACHE=/theapp/dataset_cache
ENV TRANSFORMERS_CACHE=/theapp/transformer_cache
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV MINICONDA_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
ENV CONDA=${THEAPP}/miniconda/bin/conda

RUN echo 'APT { Get { AllowUnauthenticated "1"; }; };' > /etc/apt/apt.conf.d/99allow-unauth && \
    echo America/New_York > /etc/timezone && \
    apt update && apt-get install -y build-essential bash language-pack-en-base nginx software-properties-common supervisor wget && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get install -y python3.10 python3.10-dev python3.10-venv && \
    useradd --create-home --home-dir ${THEAPP} --shell /sbin/nologin --system ${RUNNING_USER}
COPY --chown=${RUNNING_USER}:${RUNNING_USER} environment.yml ${THEAPP}/
USER ${RUNNING_USER}
RUN wget -q https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT} -O /theapp/${MINICONDA_SCRIPT} && \
    bash /theapp/${MINICONDA_SCRIPT} -b -p ${THEAPP}/miniconda && \
#    ${CONDA} update -n base -c defaults conda && \
    ${CONDA} env create --debug -f ${THEAPP}/environment.yml && \
    mkdir ${THEAPP}/endpoint
RUN ${CONDA} install pytorch torchvision cudatoolkit=11 -c pytorch
RUN ${THEAPP}/miniconda/envs/lebowski/bin/python -c 'import torch; from transformers.pipelines import pipeline; engine = pipeline("summarization", model="google/pegasus-cnn_dailymail"); engine("input")'
COPY --chown=${RUNNING_USER}:${RUNNING_USER} container-data/* ${THEAPP}/
COPY --chown=${RUNNING_USER}:${RUNNING_USER} endpoint/* ${THEAPP}/endpoint/

EXPOSE 80
EXPOSE 443

USER root
WORKDIR ${THEAPP}
CMD ["/usr/bin/supervisord", "-c", "/theapp/supervisord.ini"]
