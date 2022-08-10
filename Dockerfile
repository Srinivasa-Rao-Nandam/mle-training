FROM continuumio/miniconda3

COPY deploy/conda/env.yml regression/env/
RUN conda env update -n base -f regression/env/env.yml \
    && conda install --no-update-deps tini \
    && conda clean -afy

WORKDIR /regression/src/
COPY src/regression/* /regression/src/.
RUN mkdir logs
RUN mkdir artifacts
RUN python ingest_data.py
RUN python train.py

CMD ["python", "score.py"]