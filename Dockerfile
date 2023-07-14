FROM python:3.10.5-buster

RUN mkdir container_src

COPY sa_app/config_files/app_cfg.yml container_src/

RUN pip install sa-app

RUN python -m spacy download en_core_web_sm

WORKDIR /container_src

CMD ["python", "-m", "sa_app.app", "--config", "/container_src/app_cfg.yml"]
