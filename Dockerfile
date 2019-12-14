FROM conda/miniconda3

ENTRYPOINT []

SHELL ["/bin/bash", "-c"]

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# install packages by conda
RUN conda env create -n heroku-model-deploy -f ./environment.yml && \
    rm ./environment.yml
