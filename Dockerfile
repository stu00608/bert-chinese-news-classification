FROM huggingface/transformers-pytorch-gpu:4.23.0

WORKDIR /app
COPY . /app

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

RUN apt-get update
RUN apt-get install -y zip
RUN git clone https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset.git
RUN unzip toutiao-text-classfication-dataset/toutiao_cat_data.txt.zip -d data

RUN pip install wandb
RUN wandb login ${WANDB_API_KEY}