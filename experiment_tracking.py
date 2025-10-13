import os

import wandb

from pathlib import Path

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from types import SimpleNamespace

from utilities import get_dataloaders

from dotenv import load_dotenv, find_dotenv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = ["hero", "non-hero", "food", "spell", "side-facing"]

DATA_DIR = Path('./data/')

INPUT_SIZE = 3 * 16 * 16
HIDDEN_SIZE = 256
NUM_WORKERS = 2
OUTPUT_SIZE = 5


def get_model(dropout):
    # Simple MLP with Dropout

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.BatchNorm1d(HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    ).to(DEVICE)


# Let's define a config object to store our hyperparameters (similar to a python dictionary)
config = SimpleNamespace(
    epochs=2,
    batch_size=128,
    lr=1e-5,
    dropout=0.5,
    slice_size=10_000,
    valid_pct=0.2
)


def train_model(config):
    # Train a model with a given config

    # Initiate a W&B run
    # A run in W&B means a unit of computation
    # Generally a run corresponds to a ML experiment
    # Passing the project name and the config object
    wandb.init(
        project='sprites',
        config=config
    )

    # Get the data
    train_dl, valid_dl = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=config.batch_size,
        slice_size=config.slice_size,
        valid_pct=config.valid_pct
    )

    # A simple MLP model
    model = get_model(dropout=config.dropout)

    # Make the loss
    loss_func = nn.CrossEntropyLoss()

    # Make the optimizer
    optimizer = Adam(model.parameters(), lr=config.lr)

    example_ct = 0

    for epoch in tqdm(range(config.epochs), total=config.epochs):
        model.train()

        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            train_loss = loss_func(outputs, labels)

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

            example_ct += len(images)

            metrics = {
                'train/train_loss': train_loss,
                'train/epoch': epoch + 1,
                'train/example_ct': example_ct
            }
            # Send to Weights & Biases. Log metrics over time to visualize the performance.
            wandb.log(metrics)

        # Compute validation metrics, log images on last epoch
        val_loss, accuracy = validate_model(model=model, valid_dl=valid_dl, loss_func=loss_func)

        # Compute train and validation metrics
        val_metrics = {
            'val/val_loss': val_loss,
            'val/val_accuracy': accuracy
        }
        # Send to Weights & Biases. Log metrics over time to visualize the performance.
        wandb.log(val_metrics)


def validate_model(model, valid_dl, loss_func):
    # Compute the performance of the model on the validation dataset
    val_loss = 0.0
    correct = 0

    model.eval()

    with torch.inference_mode():
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)


# Get the api key
_ = load_dotenv(find_dotenv())
wandb_api_key = os.environ.get('WANDB_API_KEY')


if __name__ == '__main__':
    # Log in to the W & B account (possibility to connect anonymously: wandb.login(anonymous="allow"))
    wandb.login(key=wandb_api_key)

    # Train the model
    # train_model(config=config)

    """ Change parameters values """

    # So let's change the learning rate to a 1e-3
    # and see how this affects our results.
    # config.lr = 1e-3
    # train_model(config=config)

    # So let's change the number of epoch et the dropout to 0.1
    # and see how this affects our results.
    # config.dropout = 0.1
    # config.epochs = 1
    # train_model(config)

    # So let's change the learning rate to a 1e-4
    # and see how this affects our results.
    config.lr = 1e-4
    train_model(config)