# BERT Chinese text classification

* Use `ALBERT` to fine-tune on Chinese News Genre Dataset.

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
    
