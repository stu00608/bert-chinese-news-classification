# BERT Chinese text classification

* Use `ALBERT` to fine-tune on Chinese News Genre Dataset.

## Docker

* Clone and enter this repo folder first.
* You need a wandb account and get the api key.

```
export WANDB_API_KEY=<your wandb api key>
docker build --no-cache -t albert_news_classification --build-arg WANDB_API_KEY=$WANDB_API_KEY .

docker run --rm -it --gpus all albert_news_classification python3 albert_trainer.py

# or you can run this command to modify things in container.
docker run --rm -it --gpus all albert_news_classification bash
```

## Installation

* Follow pytorch official installation, to install pytorch according to your need (CPU, GPU).

```
pip install -r requirements.txt
pip install wandb

# Setup wandb for tracing training step.
wandb login
```

## Dataset

* Download Chinese News Genre Dataset and put the `.txt` file into `data/`. [Link](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)

    ```
    git clone git@github.com:aceimnorstuvwxz/toutiao-text-classfication-dataset.git
    ```

## Run

```
python albert_trainer.py
```
    
