

FROM continuumio/miniconda:4.7.12
COPY . /opt/data_prep

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

RUN conda config --set auto_activate_base false && \
conda env create -f /opt/data_prep/environment_melloddy_tuner.yml

RUN /bin/bash -c "source activate melloddy_tuner"

WORKDIR /opt/data_prep

ENV CONDA_DEFAULT_ENV melloddy_tuner
RUN conda run -n melloddy_tuner python setup.py install


ENTRYPOINT [ "/usr/bin/tini", "--" ]``
CMD "python bin/prepare_4_melloddy.py"




