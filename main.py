# main.py
import os
import torch

from config import ModelConfig
from dataloader import DataLoader
from model import GPT
from train import Trainer


def main():
    config = ModelConfig()

    # (Optional) common settings for accuracy and speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Model/optimization
    model = GPT(config=config)
    device = torch.device(config.device_type if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Call torch.compile only once
    model = torch.compile(model)

    # close to nanoGPT setting
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        ########## NEW ##########
        fused=True,  # Use fused AdamW if available for better performance. Requires PyTorch 2.0+ and compatible hardware.
        ########## NEW ##########
    )

    # DataLoader
    data_dir = os.environ.get("DATA_DIR", "/home/ubuntu/washington-filesystem") # ex: /home/ubuntu/virginia-filesystem
    data_loader = DataLoader(data_dir=data_dir, config=config)

    checkpoint_dir = os.environ.get("CKPT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.train()

if __name__ == "__main__":
    main()
