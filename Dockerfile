FROM nvcr.io/nvidia/pytorch:22.08-py3



ENV RUNNING_USER=nginx
ENV THEAPP=/theapp
ENV HF_DATASETS_CACHE=/theapp/dataset_cache
ENV TRANSFORMERS_CACHE=/theapp/transformer_cache
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN echo "APT { Get { AllowUnauthenticated "1"; }; };" > /etc/apt/apt.conf.d/99allow-unauth
RUN echo America/New_York > /etc/timezone
RUN apt update && apt-get install -y build-essential bash language-pack-en-base nginx software-properties-common supervisor && add-apt-repository ppa:deadsnakes/ppa && apt-get install -y python3.10 python3.10-dev python3.10-venv
RUN useradd --create-home --home-dir ${THEAPP} --shell /sbin/nologin --system ${RUNNING_USER}
USER ${RUNNING_USER}
RUN python3.10 -m venv ${THEAPP}/venv
COPY --chown=${RUNNING_USER}:${RUNNING_USER} poetry.lock pyproject.toml container-data/* ${THEAPP}/
RUN mkdir ${THEAPP}/endpoint
COPY --chown=${RUNNING_USER}:${RUNNING_USER} endpoint/* $THEAPP/endpoint/
WORKDIR ${THEAPP}
RUN bash -c "source venv/bin/activate && pip install --upgrade pip poetry && poetry install"

EXPOSE 80
EXPOSE 443

USER root
CMD ["/usr/bin/supervisord", "-c", "/theapp/supervisord.ini"]
