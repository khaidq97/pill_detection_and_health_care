FROM python:3.8.14-bullseye

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install torch torchvision
RUN pip install -U openmim
RUN mim install mmcv-full
RUN pip install mmcv==1.6.0
RUN pip install pythonRLSA

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD [ "python3", "run.py",  "--root=/app/data", "--save_dir=/app/output"]