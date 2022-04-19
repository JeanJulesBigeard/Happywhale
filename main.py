# General imports
import imp
import os
import gc
import cv2
import math
import copy
import time
import random
import wandb

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Import the model
from model import HappyWhaleModel

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style

b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings

warnings.filterwarnings("ignore")

# Utils
from utils import set_seed

# Dataset
from dataset import HappyWhaleDataset

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Set up wandb
wandb.init(project="HappyWhale", entity="melkus37")


# Set the config to run
CONFIG = {
    "seed": 2022,
    "epochs": 25,
    "model_name": "tf_efficientnet_b0_ns",
    "num_classes": 15587,
    "embedding_size": 512,
    "train_batch_size": 8,
    "valid_batch_size": 32,
    "learning_rate": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 1e-6,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # ArcFace Hyperparameters
    "s": 30.0,
    "m": 0.50,
    "ls_eps": 0.0,
    "easy_margin": False,
    "data_transforms": {
        "train": A.Compose(
            [
                A.Resize(448, 448),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                A.Resize(448, 448),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    },
}

OUTPUT_DIR = "output/"

# Capture a dictionary of hyperparameters with config
wandb.config = CONFIG

# Set seed for reprocibility
set_seed(CONFIG["seed"])

# Set usefull directories
ROOT_DIR = "/home/jean/datas/happy-whale-and-dolphin"
TRAIN_DIR = "/home/jean/datas/happy-whale-and-dolphin/train_images"
TEST_DIR = "/home/jean/datas/happy-whale-and-dolphin/test_images"


def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id}"


# Read the data
df = pd.read_csv(f"{ROOT_DIR}/train.csv")
df["file_path"] = df["image"].apply(get_train_file_path)

encoder = LabelEncoder()
df["individual_id"] = encoder.fit_transform(df["individual_id"])

# Create folds
skf = StratifiedKFold(n_splits=CONFIG["n_fold"])

for fold, (_, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
    df.loc[val_, "kfold"] = fold


# Data augmentations
data_transforms = CONFIG["data_transforms"]

# Create the model
model = HappyWhaleModel(CONFIG["model_name"], CONFIG["embedding_size"], CONFIG)
model.to(CONFIG["device"])

# Define the loss function
def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


# Traning function
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG["n_accumulate"]

        loss.backward()

        if (step + 1) % CONFIG["n_accumulate"] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(
            Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        )
    gc.collect()

    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch, optimizer):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(
            Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        )

    gc.collect()

    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=CONFIG["device"],
            epoch=epoch,
        )

        val_epoch_loss = valid_one_epoch(
            model,
            valid_loader,
            device=CONFIG["device"],
            epoch=epoch,
            optimizer=optimizer,
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)

        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(
                f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
            )
            best_epoch_loss = val_epoch_loss
            wandb.log({"Best Loss": best_epoch_loss})
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = os.path.join(
                OUTPUT_DIR, "Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
            )
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer):
    if CONFIG["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["T_max"], eta_min=CONFIG["min_lr"]
        )
    elif CONFIG["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CONFIG["T_0"], eta_min=CONFIG["min_lr"]
        )
    elif CONFIG["scheduler"] == None:
        return None

    return scheduler


def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = HappyWhaleDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = HappyWhaleDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG["valid_batch_size"],
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader


# Prepare dataloader
train_loader, valid_loader = prepare_loaders(df, fold=0)

# Define optimizer and scheduler
optimizer = optim.Adam(
    model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
)
scheduler = fetch_scheduler(optimizer)


model, history = run_training(
    model, optimizer, scheduler, device=CONFIG["device"], num_epochs=CONFIG["epochs"]
)
