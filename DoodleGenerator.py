import numpy as np
from doodle_parsing_utils import *
import re
import clip
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import torch.nn as nn
import torch.optim
from doodle_dataset import DoodleDataset
from doodle_predictor_config import DoodlePredictorConfig
from doodle_predictor import DoodlePredictor, calculate_loss

ENABLE_WANDB = True

if ENABLE_WANDB:
    import wandb
    wandb.login()

config = DoodlePredictorConfig("./configs/base_config.json")

if ENABLE_WANDB:
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=config.config_dict
    )

# Because we include the classname embedding as well as the stroke data, we request groups of block_size - 1
# strokes from the dataset.

mean = np.array([1.1523420567837017, -1.7892017952608004])
std = np.array([17.661761786706826, 19.484090706411575])

train_dataset = DoodleDataset(
    Path("./dataset"),
    split="train",
    block_size=config.block_size - 1,
    scaled_size=config.scaled_size,
    device=config.device,
    mean=mean, # Normalization
    std=std,
)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)

test_dataset = DoodleDataset(
    Path("./dataset"),
    split="test",
    block_size=config.block_size - 1,
    scaled_size=config.scaled_size,
    device=config.device,
    mean=mean,  # Normalization
    std=std,
)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

doodle_predictor = DoodlePredictor(config).to(config.device)
optimizer = torch.optim.Adam(doodle_predictor.parameters(), lr=config.lr)

num_epochs = config.num_epochs
train_losses = []

for epoch_num in tqdm(range(num_epochs)):
    for batch in tqdm(train_dataloader):
        xs, ys, ex_classname_embeddings = batch
        xs, ys, ex_classname_embeddings = (
            xs.to(config.device),
            ys.to(config.device),
            ex_classname_embeddings.to(config.device),
        )

        model_out = doodle_predictor(xs.float(), ex_classname_embeddings.float())

        loss = calculate_loss(
            model_out.double(), ys, position_coeff=config.position_loss_coeff, pen_state_coeff=config.pen_state_loss_coeff
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ENABLE_WANDB:
            wandb.log({"train_loss" : loss.item()})
    
    test_losses = []
    doodle_predictor.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            xs, ys, ex_classname_embeddings = batch
            xs, ys, ex_classname_embeddings = (
                xs.to(config.device),
                ys.to(config.device), 
                ex_classname_embeddings.to(config.device),
            )
            model_out = doodle_predictor(xs.float(), ex_classname_embeddings.float())
            
            test_loss = calculate_loss(
                model_out.double(), ys, position_coeff=config.position_loss_coeff, pen_state_coeff=config.pen_state_loss_coeff
            )
            test_losses.append(test_loss.item())
    
    if ENABLE_WANDB:
        avg_test_loss = sum(test_losses) / len(test_losses)
        wandb.log({"epoch": epoch_num, "avg_test_loss": avg_test_loss})
    
    doodle_predictor.train()

if ENABLE_WANDB:
    wandb.finish()

os.makedirs(config.output_directory, exist_ok=True)
torch.save(doodle_predictor.state_dict(), f"./{config.output_directory}/model_state_dict")
with open(f"./{config.output_directory}/config.json", "w") as f:
    config_dict = config.config_dict
    json.dump(config_dict, f)
