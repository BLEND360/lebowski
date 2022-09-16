FROM --platform=linux/amd64 nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV RUNNING_USER=nginx
ENV THEAPP=/theapp
ENV HF_DATASETS_CACHE=/theapp/dataset_cache
ENV TRANSFORMERS_CACHE=/theapp/transformer_cache
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV MINICONDA_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
ENV CONDA=${THEAPP}/miniconda/bin/conda
ARG tz=America/New_York
ENV _TIMEZONE=${tz}
ARG pytorch_suffix
ENV _PYTORCH_SUFFIX=${pytorch_suffix}
ARG conda_debug=--debug
ENV _CONDA_DEBUG=${conda_debug}

RUN echo 'APT { Get { AllowUnauthenticated "1"; }; };' > /etc/apt/apt.conf.d/99allow-unauth && \
    echo ${_TIMEZONE} > /etc/timezone && \
    apt update && \
    apt-get install -y build-essential bash language-pack-en-base nginx software-properties-common supervisor wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv && \
    useradd --create-home --home-dir ${THEAPP} --shell /sbin/nologin --system ${RUNNING_USER}
COPY --chown=${RUNNING_USER}:${RUNNING_USER} environment.yml ${THEAPP}/
USER ${RUNNING_USER}
RUN wget -q https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT} -O /theapp/${MINICONDA_SCRIPT} && \
    bash /theapp/${MINICONDA_SCRIPT} -b -p ${THEAPP}/miniconda && \
    ${CONDA} env create ${_CONDA_DEBUG} -f ${THEAPP}/environment.yml && \
    mkdir ${THEAPP}/endpoint
RUN ${CONDA} install pytorch torchvision cudatoolkit=11 -c pytorch${_PYTORCH_SUFFIX}
COPY --chown=${RUNNING_USER}:${RUNNING_USER} test.py ${THEAPP}/
RUN ${THEAPP}/miniconda/envs/lebowski/bin/python ${THEAPP}/test.py
COPY --chown=${RUNNING_USER}:${RUNNING_USER} container-data/* ${THEAPP}/
ADD --chown=${RUNNING_USER}:${RUNNING_USER} endpoint/ ${THEAPP}/endpoint/

EXPOSE 80
EXPOSE 443

USER root
ARG n_threads=1
ENV N_THREADS=${n_threads}
RUN sed -re "s/threads =.*/threads = ${N_THREADS}/" -i ${THEAPP}/uwsgi.ini
ENV AWS_DEFAULT_REGION=us-east-1
WORKDIR ${THEAPP}
CMD ["/usr/bin/supervisord", "-c", "/theapp/supervisord.ini"]
