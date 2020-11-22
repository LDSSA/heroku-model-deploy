FROM python:3.8-buster

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# install packages by conda
RUN pip install -r requirements_prod.txt
CMD ["python", "app.py"]
