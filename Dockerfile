FROM python:3.10.5-buster

RUN mkdir container_src

COPY sa_app/config_files/app_cfg.yml container_src/

RUN pip install sa-app==0.0.1

RUN python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["python", "-m", "sa_app.app", "--config", "/container_src/app_cfg.yml"]
