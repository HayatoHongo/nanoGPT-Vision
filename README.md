# nanoGPTVision

**ChatGPT-4o–like Vision AI that $100 can buy.**

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

- Model: GPT-style language model
- Hardware: **Lambda Cloud A100 × 8**
- Time: **~6 hours**
- Cost: **~$90**



---

### 2. Language SFT
- Vision encoder: **OpenAI CLIP**
- Hardware: **Google Colab Pro – A100 (high memory)**
- Time: **~5 hours**
- Cost: **~$4**

Available on Colab!

https://colab.research.google.com/drive/1CvgpTAJzpsZjraCSxJ8phYyAmSMw-LwJ?usp=sharing

---

### 3. Vision pretraining (for the SFT model)
- Hardware: **Google Colab Pro – A100 (high memory)**
- Time: **~3 hours**
- Cost: **~$2**



### 4.  Vision pretraining (for the Vision Pretrained model)

Still under construction! please wait! we will release in January!

---

### 💰 Total cost
**≈ $96 USD**

---

## Current status

- [x] From-scratch text decoder
- [x] CLIP-based vision encoder
- [x] Vision-language pretraining
- [ ] Expanded multimodal instruction tuning (WIP)
- [ ] Code cleanup & documentation (in progress)

## Acknowledgements

- **Andrej Karpathy** — nanoGPT and its philosophy
- OpenAI — CLIP


This repository is provided for research and educational purposes.
Expect rough edges, missing pieces, and ongoing refactors.
