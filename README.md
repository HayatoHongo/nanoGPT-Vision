# nanoGPT-Vision

> ChatGPT4o–like Vision AI that $100 can buy.

![app_chat.png](app_chat.png)

## [▶️ Link to WebUI](https://huggingface.co/spaces/HayatoHongoEveryonesAI/EveryonesGPT_Vision_Pretrained)

⚠️ **Status: actively being cleaned up and documented.**
The core ideas and training runs are real and reproducible,  
but parts of the codebase and README are still under active refinement.

---

`nanoGPTVision` is a **minimal Vision-Language Model built end-to-end**,  
featuring a **GPT-style text decoder trained fully from scratch**.

The project is heavily inspired by **Andrej Karpathy’s nanoGPT** and follows the same philosophy:
> understand the system by building it yourself.

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
this is the first publicly released publicly to build vision language model from scratch,
LLM and vision projector.


---

## Project philosophy

This is **not a polished product**.

This is a **work-in-progress research and learning project**.

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
- Vision encoder: **OpenAI CLIP**
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

Still under construction! please wait! we will release in January!



---

### 💰 Total cost
**≈ $96 USD**

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
- GQA: It does not only help training speed but also increase loss. KV cache matters for >10k tokens inference. It is not considered.
- MLA (in DeepSeek-V3): It does not help training speed. Even during inference, it is incompatible with SDPA. KV cache matters for >10k tokens inference. It is not considered.

[Minmind](https://github.com/jingyaogong/minimind) provided on/off switch for those techniques, which greatly helped me to understand the diferrences.


## Acknowledgements

- Andrej Karpathy — nanoGPT, nanoChat(for streaming inference) and its philosophy
- OpenAI — CLIP
- Haotian Liu - LLaVA
- Sebastian Raschka - LLM SFT 
- HuggingFace Team - FineWeb-Edu Dataset
- ShallowU - FineWeb-Edu npy dataset
- jingyaogong - Minimind project https://github.com/jingyaogong/minimind

This repository is provided for research and educational purposes.
Expect rough edges, missing pieces, and ongoing refactors.
