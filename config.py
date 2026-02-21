import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # === training ===
    # Single-device setup: global batch size == batch_size
    batch_size: int = 16 # 16 is equivalent to about 25 GB VRAM usage. 
    total_training_steps: int = 200_000
    evaluation_frequency: int = 100
    checkpoint_save_frequency: int = 10_000
    evaluation_loops: int = 10

    # === sequence ===
    input_sequence_length: int = 1024
    max_sequence_length: int = 2048

    # === model ===
    embedding_dim: int = 1280 # 384
    hidden_dim: int = 5120 # 1536
    num_attention_heads: int = 10 # 6
    layer_count: int = 20
    """DELETE
    rope_theta: float = 1_000_000.0
    """
    ### NEW ###
    num_relative_positions = 256
    vocab_size: int = 50257
    ### NEW ###

    # === optimization ===
    # The learning rate is VERY important. You can tune it, but that takes time, money, and a bit of your sanity.
    # The "best" value depends on the architecture, the batch size, global batch size and more.
    # In short: nobody truly knows the best learning rate for *your* model. Just try different values.
    # As for this ~500MB model, with batch size 256 (A100x8 setting), I found:
    # max lr: 1e-2: loss diverges. 1e-3 : good, 1e-4 : too slow compared to 1e-3
    # This time, with batch size 48, learning rate is scaled down by about 5x from 256 batch size setting.
    
    max_learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    warmup_steps: int = 1_000

    # === system ===
    device_type: str = "cuda"
    random_seed_value: int = 1337
    autocast_dtype: torch.dtype = torch.bfloat16
