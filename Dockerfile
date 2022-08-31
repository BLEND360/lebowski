FROM python:3.10-bullseye

ENV RUNNING_USER=nginx
ENV THEAPP=/theapp

ENV HF_DATASETS_CACHE=/theapp/dataset_cache
ENV TRANSFORMERS_CACHE=/theapp/transformer_cache
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && apt-get -y upgrade && apt-get install -y build-essential bash nginx supervisor
RUN useradd --create-home --home-dir ${THEAPP} --shell /sbin/nologin --system ${RUNNING_USER}
USER ${RUNNING_USER}
RUN python -m venv ${THEAPP}/venv
COPY --chown=${RUNNING_USER}:${RUNNING_USER} poetry.lock pyproject.toml container-data/* ${THEAPP}/
RUN mkdir ${THEAPP}/endpoint
COPY --chown=${RUNNING_USER}:${RUNNING_USER} endpoint/* $THEAPP/endpoint/
WORKDIR ${THEAPP}
RUN bash -c "source venv/bin/activate && pip install --upgrade pip poetry && poetry install"

EXPOSE 80
EXPOSE 443

USER root
CMD ["/usr/bin/supervisord", "-c", "/theapp/supervisord.ini"]
