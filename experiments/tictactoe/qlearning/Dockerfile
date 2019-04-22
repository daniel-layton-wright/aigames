# Install the latest version of pytorch
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
WORKDIR /root

# Installs pandas, google-cloud-storage, and cloudml-hypertune
RUN pip install pandas google-cloud-storage cloudml-hypertune
RUN pip install git+https://github.com/daniel-layton-wright/aigames

# Copies the trainer code to the docker image.
COPY train.py ./train.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "train.py"]