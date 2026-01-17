# nanoGPT-Vision

> ChatGPT4o–like Vision AI that $100 can buy.

![Demo](assets/demo.gif)


## [▶️ Link to WebUI](https://huggingface.co/spaces/HayatoHongoEveryonesAI/EveryonesGPT_Vision_Instruct)

⚠️ **Status: actively being cleaned up and documented.**
The core ideas and training runs are real and reproducible,  
but parts of the codebase and README are still under active refinement.

---

`nanoGPTVision` is a **minimal Vision-Language Model built end-to-end**,  
featuring a **GPT-style text decoder trained fully from scratch**.

A pretrained **CLIP vision encoder** is used for visual representations (Sorry😉),  
while **all language modeling and vision–language training code is written, data is public, and trained from scratch**.

---

## Why this project exists

Most open VLMs reuse large pretrained language models (LLaMA, Vicuna, etc.).

**nanoGPTVision does not.**

Instead, it focuses on:
- training the **text decoder from scratch**
- keeping the architecture minimal and readable
- making the design choices explicit
- showing how far you can go on a small, transparent budget

Alghough the CLIP encoder is external, to the best of our knowledge, 
this is the first educational project to pretrain both text decoder and vision projector from scratch.

---

## 😀 Can I learn nanoGPT before nanoGPT-Vision?

If you have not implemented nanoGPT yet, Learn on Colab!

**Free T4 GPU on colab!😊**

[Everyones_nanoGPT](https://github.com/HayatoHongo/Everyones_nanoGPT/tree/main)

| Chapter  | Estimated Time | English 🇺🇸 |
|---|---|---|
| Chapter 00: Start Tutorial      | 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter00_todo.ipynb) |
| Chapter 01: Dataloader         | 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter01_todo.ipynb) |
| Chapter 02: TokenEmbedding     | 0.5-1 hour  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter02_todo.ipynb) |
| Chapter 03: PositionEmbedding  | 0.5-1 hour  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter03_todo.ipynb) |
| Chapter 04: EmbeddingModule    | 0.5-1 hour  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter04_todo.ipynb) |
| Chapter 05: LayerNorm          | 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter05_todo.ipynb) |
| Chapter 06: AttentionHead      | 3-4 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter06_todo.ipynb) |
| Chapter 07: MultiHeadAttention | 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter07_todo.ipynb) |
| Chapter 08: FeedForward        | 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter08_todo.ipynb) |
| Chapter 09: TransformerBlock   | 0.5-1 hour  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter09_todo.ipynb) |
| Chapter 10: VocabularyLogits   | 0.5-1 hour  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter10_todo.ipynb) |
| Chapter 11: nanoGPT| 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter11_todo.ipynb) |
| Chapter 12: Trainer            | 1-2 hour    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/main/notebooks/todo/Everyones_nanoGPT_colab_Chapter12_todo.ipynb) |

Sorry the following is in Japanes. Translation is undergoing!

| チャプター  | 推定所要時間 | ノートブック  |
|---|---|---|
| Chapter 13: Tokens per second(CPU)    | 1~2時間 | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter13_todo_ja.ipynb) |          |
| Chapter 14: Tokens per second(T4 GPU)     | 0.5〜1時間 | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter14_todo_ja.ipynb) |          |
| Chapter 15: Train nanoGPT with GPU    | 0.5〜1時間    | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter15_todo_ja.ipynb) |          |
| Chapter 16: モデルサイズだけ大きくする          | 0.5 ~ 1 時間 (+ モデル学習 1時間)  | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter16_todo_ja.ipynb) |          |
| Chapter 17:  データセットを大きくする    | 1〜2時間 (+ モデル学習 1時間) | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter17_todo_ja.ipynb) |          |
| Chapter 18: tiktoken      | 1〜2時間 (+ モデル学習 1時間)   | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter18_todo_ja.ipynb) |          |
| Chapter 19: Long Train    | 1〜2時間 (+ モデル学習 **6時間** セッション切れ工夫必要)  | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter19_todo_ja.ipynb) |          |
| Chapter 20: 学習率            | 0.5〜1時間   | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter20_todo_ja.ipynb) |          |
| Chapter 21: Scaling Law       | 1〜2時間 | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter21_todo_ja.ipynb) |          |
| Chapter 22: TinyStories(メイン) | 1〜2時間   | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter22_main_todo_ja.ipynb) |          |
| Chapter 22: TinyStories(モデル学習) | 1時間   | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter22_train_todo_ja.ipynb) |          |
| Chapter 23: RPE(OverSimplified) | 2~3時間   | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter23_todo_ja.ipynb) |          |
| Chapter 24: RPE(Simplified)        | 1〜2時間 (+ モデル学習 1時間)      | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HayatoHongo/Everyones_nanoGPT/blob/ja/notebooks/todo/Everyones_nanoGPT_colab_Chapter24_todo_ja.ipynb) |


Waiting List

RoPE, SDPA, LR Schedule, Checkpoint. 

---

## Training setup & cost breakdown

All numbers below are **actual runs**, not estimates.

# How to build your own Vision Language Model

### 1. Language Pretraining

- Model: GPT-style language model
- Hardware: **Lambda Cloud A100 × 8**
- Time: **~6 hours**
- Cost: **~$90**

#### Create HuggingFace Account

[44:18 ~ 46:18 Hands On Video on how to create HuggingFace account](https://youtu.be/qkS_Zc6uvbo?si=JWKHgWKlX4_Qw2Vk)

Create Access Tokens

https://huggingface.co/settings/tokens

Publish Fine-Grained token. Mark all checkpoints on Repository.

#### Use Lambda Cloud

[Lambda Cloud](https://cloud.lambda.ai/instances)

For early birds who try SSH for the first time, this might be the biggest challenge.

Make sure you select Ubuntu 22.04.

[~ 9:30 Hands On Video on how to use Lambda Cloud SSH](https://youtu.be/qkS_Zc6uvbo?si=JWKHgWKlX4_Qw2Vk)

- Just watch the first 10 minutes, the later part is about nanoGPT, not this one. (But nanoGPT is also great!)

```bash
git clone https://github.com/HayatoHongo/nanoGPTVision.git
cd nanoGPTVision
```

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
pip install -U huggingface_hub
```

```bash
pip install torch numpy datasets tiktoken
```

```bash
pip install huggingface_hub
```

Replace YOURFILESYSTEM.

```bash
python3 - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ShallowU/FineWeb-Edu-10B-Tokens-NPY",
    repo_type="dataset",
    local_dir="/home/ubuntu/YOURFILESYSTEM",
    local_dir_use_symlinks=False,
)
EOF
```

@main.py
Replace YOURFILESYSTEM

train
```bash
torchrun --standalone --nproc_per_node=8 main.py
```

Upload to HuggingFace

```bash
export HF_TOKEN="hf_xxx.....xx"
```

```bash
echo $HF_TOKEN
```

@upload.py
Replace YOURNAME, YOUREPO, YOURFILESYSTEM

You don't need to make repository in advance.

```bash
python upload.py
```

## WebUI

Please clone this huggingface space and replace the model checkpoint with your one.

https://huggingface.co/spaces/HayatoHongoEveryonesAI/EveryonesGPT_Pretrained

---

### 2. Language SFT
- Hardware: **Google Colab Pro – A100 (high memory)**
- Time: **~5 hours**
- Cost: **~$4**

Available on Colab!

https://colab.research.google.com/drive/1CvgpTAJzpsZjraCSxJ8phYyAmSMw-LwJ?usp=sharing


## WebUI

Please clone this huggingface space and replace the model with your one.

https://huggingface.co/spaces/HayatoHongoEveryonesAI/EveryonesGPT_SFT

---

### 3. Vision pretraining (for the SFT model)
- Hardware: **Google Colab Pro – A100 (high memory)**
- Time: **~3 hours**
- Cost: **~$2**

Available on Colab!

https://colab.research.google.com/drive/1QR8ygk2RsGuDt9w8Nmz0Zn8aC6MBYoD6?usp=sharing


#### Inference

Available on Colab!

https://colab.research.google.com/drive/1GK9y0BAt2Xdyploc5B55kyNlxyZuwckQ?usp=sharing

#### Web UI

Please clone this huggingface space and replace the model checkpoint with your one.

https://huggingface.co/spaces/HayatoHongoEveryonesAI/EveryonesGPT_Vision_Pretrained



### 4.  Vision Instruction Tuning (for the Vision Pretrained model)
- Hardware: **Google Colab Pro – A100 (high memory)**
- Time: **~2 hours**
- Cost: **~$1~2**

Available on Colab!

https://colab.research.google.com/drive/1FTstgyIWpi-VY0Slylcj4s_Bivojo6xF?usp=sharing

#### Inference

Available on Colab!

https://colab.research.google.com/drive/13no1R7vexor0UJSp_Wr6eB0xKZBDOOxY?usp=sharing

#### Web UI

Please clone this huggingface space and replace the model checkpoint with your one.

https://huggingface.co/spaces/HayatoHongoEveryonesAI/EveryonesGPT_Vision_Instruct


---

### 💰 Total cost
**≈ $97~98 USD**

---

## Current status

- [x] From-scratch text decoder
- [x] CLIP-based vision encoder
- [x] Vision-language pretraining
- [ ] Expanded Vision instruction tuning (WIP)
- [ ] Code cleanup & documentation (in progress)
- [ ] Build Clip from scratch


## What we did not include in model.py 

Recently there are so many techniques to boost LLM training that we don't know which is critical.

I removed uncritical techniques  and only critical technique remaned.

We included 

- RoPE: Traditional RPE(like RPE in T5) worked slighty worse than RoPE in my own prior experiments with smaller model. But the biggest problem about RPE is that it is incompatible with Scaled Dot Product Attention.
- Scaled Dot Product Attention (Flash Attention): 😉 Sorry I don't understand the inner machanism of that. But it does not meddle in model, just increase training speed significantly and purely.


We excluded these techniques as uncritical

- RMSNorm: normal LayerNorm is enough.
- SwiGLU(and GeLU): normal ReLU is enough.
- Vocab tying: uncritical
- GQA: It does not help training speed. Moreover it also increase loss. KV cache matters for >10k tokens inference, which is not the case for this project.
- MLA (in DeepSeek-V3): It does not help training speed. Even during inference, it is incompatible with SDPA. KV cache matters for >10k tokens inference, which is not the case for this project.

[Minimind](https://github.com/jingyaogong/minimind) provided on/off switch for those techniques, which greatly helped me to understand the diferrences.


## Acknowledgements

# Dataset
- HuggingFace Team - FineWeb-Edu Dataset https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- ShallowU - FineWeb-Edu Dataset in numpy format https://huggingface.co/datasets/ShallowU/FineWeb-Edu-10B-Tokens-NPY
- Haotian Liu - LLaVA Pretrain Dataset (whitelisted version was used) https://huggingface.co/datasets/HayatoHongo/LLaVA-CC3M-Pretrain-521K
- Haotian Liu - LLaVA Instruction Dataset https://huggingface.co/datasets/HayatoHongo/LLaVA-Instruct-150K/tree/main
- Magpie Team - https://huggingface.co/datasets/Magpie-Align/Magpie-Phi3-Pro-1M-v0.1
- roneneldan - https://huggingface.co/datasets/roneneldan/TinyStories

# Code
- Andrej Karpathy — nanoGPT, nanoChat(for streaming inference) and its philosophy https://github.com/karpathy/nanoGPT
- OpenAI — CLIP - https://huggingface.co/openai/clip-vit-large-patch14
- Haotian Liu - LLaVA https://github.com/haotian-liu/LLaVA
- Sebastian Raschka - LLM SFT https://github.com/rasbt/LLMs-from-scratch 
- jingyaogong - Minimind project https://github.com/jingyaogong/minimind

This repository is provided for research and educational purposes.
Expect rough edges, missing pieces, and ongoing refactors.
