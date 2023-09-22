FROM python:3.10.5-buster

RUN mkdir container_src

COPY sa_app/mongo_service/*.py container_src/

COPY sa_app/mongo_service/app_cfg.yml container_src/

COPY sa_app/mongo_service/requirements.txt container_src/

RUN pip install -r container_src/requirements.txt


EXPOSE 5002

CMD ["python", "container_src/app.py"]
