# main.py
# torchrun --standalone --nproc_per_node=8 main.py
# torchrun --standalone --nproc_per_node=1 main.py
import os
import torch
import torch.distributed as dist

from config import ModelConfig
from dataloader import DataLoader
from model import GPT
from train import Trainer, cleanup_ddp

def setup_ddp():
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    config = ModelConfig()

    # DDP init（ここで rank/world が確定）
    local_rank = setup_ddp()

    # （任意）精度と速度の定番
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # モデル/最適化
    model = GPT(config=config)

    ### NEW ###
    model = model.to(local_rank)
    model = torch.compile(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
    )
    ### NEW ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # DataLoader（DDP init後なので dist.get_rank() が使える）
    data_dir = os.environ.get("DATA_DIR", "/home/ubuntu/YOURFILESYSTEM") # os.environ.get("DATA_DIR", "/home/ubuntu/virginia-filesystem")
    data_loader = DataLoader(data_dir=data_dir, config=config)

    checkpoint_dir = os.environ.get("CKPT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        local_rank=local_rank,
    )

    try:
        trainer.train()
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()
