# dataloader.py
# refactored dataloader with train/val separation and DDP support

import os
import torch
import numpy as np


class DataLoader:
    def __init__(self, data_dir, config):
        """
        シャード化されたデータを順番に読むデータローダー。

        ※ shard という言葉を使っているが、
        ※ 実体は「分割された .npy ファイルのパス」のリストにすぎない。
        """
        self.config = config
        self.data_dir = data_dir

        
        import torch.distributed as dist
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        

        # =========================================
        # shard（= 分割されたデータファイルのパス）
        # =========================================

        self.train_shard_paths = [
            os.path.join(data_dir, f"edufineweb_train_{i:06d}.npy") for i in range(1, 100)
        ]

        self.val_shard_paths = [
            os.path.join(data_dir, "edufineweb_val_000000.npy")
        ]

        # =========================================
        # train 用の読み取り状態
        # =========================================

        self.train_shard_index = 0
        self.train_shard_tokens = self.load_shard(self.train_shard_paths[self.train_shard_index])

        """ DELETE code
        self.train_read_position = 0
        """
        
        self.train_read_position = (
            self.rank
            * self.config.batch_size
            * self.config.input_sequence_length
        )
        

        # =========================================
        # validation 用の読み取り状態
        # =========================================

        self.val_shard_index = 0
        self.val_shard_tokens = self.load_shard(self.val_shard_paths[self.val_shard_index])

        """ DELETE code
        self.val_read_position = 0
        """
        
        self.val_read_position = (
            self.rank
            * self.config.batch_size
            * self.config.input_sequence_length
        )
        

    def load_shard(self, shard_path):
        """
        shard（= 1つの .npy ファイル）を読み込み、
        torch.Tensor に変換する。
        """
        tokens_np = np.load(shard_path).astype(np.int32)
        return torch.tensor(tokens_np, dtype=torch.long)

    def get_batch(self, split):
        """
        指定された split ('train' or 'val') から
        次のバッチを順番に取り出す。
        """
        batch_size = self.config.batch_size
        sequence_length = self.config.input_sequence_length

        # -----------------------------------------
        # train
        # -----------------------------------------
        if split == "train":
            chunk = self.train_shard_tokens[
                self.train_read_position :
                self.train_read_position + batch_size * sequence_length + 1
            ]

            input_sequences = chunk[:-1].view(batch_size, sequence_length)
            target_sequences = chunk[1:].view(batch_size, sequence_length)

            """ DELETE code
            self.train_read_position += batch_size * sequence_length
            """
            
            self.train_read_position += (batch_size * sequence_length * self.world_size)
            

            # 今のシャードに次回のバッチの余裕がなくなれば次のシャードへ
            if (self.train_read_position + batch_size * sequence_length * self.world_size + 1 
                > len(self.train_shard_tokens)):
                self.train_shard_index = (self.train_shard_index + 1) % len(self.train_shard_paths)
                self.train_shard_tokens = self.load_shard(self.train_shard_paths[self.train_shard_index])

                
                self.train_read_position = (self.rank * batch_size * sequence_length)
                

        # -----------------------------------------
        # validation
        # -----------------------------------------
        elif split == "val":
            chunk = self.val_shard_tokens[
                self.val_read_position :
                self.val_read_position + batch_size * sequence_length + 1
            ]

            input_sequences = chunk[:-1].view(batch_size, sequence_length)
            target_sequences = chunk[1:].view(batch_size, sequence_length)

            """ DELETE code
            self.val_read_position += batch_size * sequence_length
            """
            
            self.val_read_position += (batch_size * sequence_length * self.world_size)
            

            # 今のシャードに次回のバッチの余裕がなくなれば次のシャードへ
            if (self.val_read_position + batch_size * sequence_length * self.world_size + 1
                > len(self.val_shard_tokens)):
                self.val_shard_index = (self.val_shard_index + 1) % len(self.val_shard_paths)
                self.val_shard_tokens = self.load_shard(self.val_shard_paths[self.val_shard_index])

                
                self.val_read_position = (self.rank * batch_size * sequence_length)
                

        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # -----------------------------------------
        # バッチを返す
        # -----------------------------------------
        return (
            input_sequences.to(self.config.device_type),
            target_sequences.to(self.config.device_type),
        )
