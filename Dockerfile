FROM continuumio/anaconda

RUN apt-get update

#ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y gcc python-dev
RUN apt-get install -y libblas-dev liblapack-dev libopenblas-dev g++

COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt

COPY . /usr/src/app/

CMD ["python run.py"]
