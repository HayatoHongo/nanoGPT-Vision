# train.py
# torch.compile を追加
# refactored dataloader with train/val separation and DDP support
import math

def get_learning_rate(current_step, config):
    max_learning_rate = config.max_learning_rate
    min_learning_rate = config.min_learning_rate
    warmup_steps = config.warmup_steps
    total_training_steps = config.total_training_steps

    if current_step < warmup_steps:
        # --- Linear Warmup ---
        warmup_progress_ratio = current_step / warmup_steps
        learning_rate = max_learning_rate * warmup_progress_ratio

    else:
        # --- Cosine Decay ---
        decay_step_index = current_step - warmup_steps
        decay_total_steps = total_training_steps - warmup_steps
        decay_progress_ratio = decay_step_index / decay_total_steps

        cosine_decay_value = math.cos(math.pi * decay_progress_ratio)
        cosine_decay_ratio = 0.5 * (1.0 + cosine_decay_value)

        learning_rate_range = max_learning_rate - min_learning_rate
        learning_rate = min_learning_rate + cosine_decay_ratio * learning_rate_range

    return learning_rate


import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, model, optimizer, data_loader, config, checkpoint_dir, local_rank):

        """ DELETE
        # DDP 初期化（single-node A100x8: torchrun --standalone --nproc_per_node=8 train.py）
        self.local_rank = setup_ddp()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main_process = self.rank == 0
        """

        ### NEW ###
        self.local_rank = local_rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main_process = self.rank == 0
        ### NEW ###

        """ DELETE
        # model を rank の GPU に載せてから compile -> DDP
        model = model.to(self.local_rank)
        model = torch.compile(model)
        self.model = DDP(model, device_ids=[self.local_rank])
        """

        ### NEW ###
        # main.py 側で DDP 済み
        self.model = model
        ### NEW ###

        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config
        self.start_step = 0
        self.checkpoint_dir = checkpoint_dir

        self.steps = []
        self.learning_rates = []
        self.train_losses = []
        self.val_losses = []
        self.tokens_per_second_list = []
        self.total_seen_tokens_list = []
        self.total_train_time_list = []


    def save_checkpoint(self, current_step):
        
        if not self.is_main_process:
            return
        

        checkpoint_data = {
            "current_step": current_step,
            
            # DDP の場合は module 側が実体
            "model_state_dict": self.model.module.state_dict(),
            
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{current_step:06d}.pt"
        )

        torch.save(checkpoint_data, checkpoint_path)
        print(f"[INFO] Successfully saved checkpoint at step {current_step:06d}")


    def load_checkpoint(self, checkpoint_path):
        """ DELETE
        checkpoint_data = torch.load(checkpoint_path, map_location=self.config.device_type)
        """
        
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location=f"cuda:{self.local_rank}" # DDP 対応
        )
        

        
        self.model.module.load_state_dict(checkpoint_data["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.start_step = checkpoint_data["current_step"] + 1
        
        if self.is_main_process:
            print(f"[INFO] Resume. Loaded checkpoint from step {checkpoint_data['current_step']}")
        

    def update_learning_rate(self, current_step):
        learning_rate = get_learning_rate(current_step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate


    """ DELETE
    def train_step(self):
        input_batch, target_batch = self.data_loader.get_batch('train')
        self.optimizer.zero_grad()
    """

    def train_step(self):
        input_batch, target_batch = self.data_loader.get_batch('train')
        self.optimizer.zero_grad()

        
        # A100 前提で bf16 autocast
        with torch.autocast(device_type=self.config.device_type, dtype=self.config.autocast_dtype):
            logits, loss = self.model(input_batch, target_batch)

        loss.backward()
        self.optimizer.step()

        
        # DDP でも安全なように detach
        return loss.detach().item()
        

    def evaluate(self):
        
        # 評価は全 rank で同じ回数実行し、all_reduce で平均を取る
        # （barrier不要：全プロセス同じ分岐・同じループ回数を通る前提）
        with torch.autocast(device_type=self.config.device_type, dtype=self.config.autocast_dtype):
        
            self.model.eval()  # 評価モードに切り替え
            losses = {"train": [], "val": []} # 学習・検証データ両方の損失を計算
            with torch.no_grad():
                for split in ['train', 'val']:
                    for _ in range(self.config.evaluation_loops):
                        input_batch, target_batch = self.data_loader.get_batch(split)
                        _, loss = self.model(input_batch, target_batch)
                        """ DELETE
                        losses[split].append(loss.item())
                        """
                        
                        losses[split].append(loss.detach())
                        
            self.model.train()  # 再び学習モードへ戻す

        """ DELETE
        # 各データセット（train, val）での損失の平均を計算して返す
        return {split: sum(values) / len(values) for split, values in losses.items()}
        """
        
        # DDP 平均（world_size で平均化）
        out = {}
        for split in ['train', 'val']:
            stacked = torch.stack(losses[split])  # (evaluation_loops,)
            mean_loss = stacked.mean()
            if dist.is_initialized():
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss = mean_loss / self.world_size
            out[split] = mean_loss.item()
        return out
        

    def train(self):
        """DELETE
        last_eval_end_time = None
        total_train_time = 0.0
        """
        
        # 時間計測・ログは main プロセスだけが責任を持つ（DDP的に一貫）
        if self.is_main_process:
            last_eval_end_time = None
            total_train_time = 0.0
        

        # (configで指定された回数+1)だけtrain_stepを実行する。
        for step in range(self.start_step, self.config.total_training_steps + 1):
            # 1回の学習ステップ（毎回行う主な処理）
            # lrをupdateする
            self.update_learning_rate(step)
            train_loss = self.train_step()

            # checkpoint 保存は main プロセスのみ（save_checkpoint内でもガード済みだが明示）
            
            if self.is_main_process:
            
                if step > 0 and step % self.config.checkpoint_save_frequency == 0:
                    self.save_checkpoint(step)

            # evaluation_frequency ごとに評価する。
            if step % self.config.evaluation_frequency == 0:
                # evaluate は all_reduce を使うので全 rank が必ず参加（安全）
                eval_loss = self.evaluate()

                
                if self.is_main_process:
                
                    if last_eval_end_time is None:  # step==0とチェックポイント再開時
                        tokens_per_second = None
                    else:  # last_eval_end_time が記録されていれば、tokens/sを計算する
                        current_eval_start_time = time.time()
                        evaluation_interval = current_eval_start_time - last_eval_end_time
                        total_train_time += evaluation_interval

                        """DELETE
                        tokens_per_evaluation_interval = self.config.batch_size * self.config.input_sequence_length * self.config.evaluation_frequency
                        """
                        
                        # グローバルに処理した token 数（DDP）
                        tokens_per_evaluation_interval = (
                            self.config.batch_size
                            * self.config.input_sequence_length
                            * self.config.evaluation_frequency
                            * self.world_size
                        )
                        

                        tokens_per_second = tokens_per_evaluation_interval / evaluation_interval

                    """DELETE
                    total_seen_tokens = self.config.batch_size * self.config.input_sequence_length * step
                    """
                    
                    total_seen_tokens = (
                        self.config.batch_size
                        * self.config.input_sequence_length
                        * step
                        * self.world_size
                    )
                    

                    current_learning_rate = self.optimizer.param_groups[0]["lr"]

                    print(
                        f"step {step:05d} | "
                        f"lr {current_learning_rate:.6e} | "
                        f"train loss {eval_loss['train']:.4f} | "
                        f"val loss {eval_loss['val']:.4f} | "
                        f"tok/s {int(tokens_per_second) if tokens_per_second is not None else 'None'} | "
                        f"tokens {total_seen_tokens:,} | "
                        f"time {total_train_time:.2f}s"
                    )

                    self.steps.append(step)
                    self.learning_rates.append(current_learning_rate)
                    self.train_losses.append(eval_loss['train'])
                    self.val_losses.append(eval_loss['val'])
                    self.tokens_per_second_list.append(tokens_per_second)
                    self.total_seen_tokens_list.append(total_seen_tokens)
                    self.total_train_time_list.append(total_train_time)

                    # この評価が終わった時間を記録する。次の評価開始時との時間差が`evaluation_interval`となる。
                    last_eval_end_time = time.time()

        # 学習が最後まで正常に終わった場合の最終モデル保存
        
        if self.is_main_process:
        
            self.save_checkpoint(self.config.total_training_steps)
