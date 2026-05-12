# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

# WandB report - https://wandb.ai/da25s006-indian-institute-of-technology-madras/da6401-a3/reports/DA6401-Assignment-3--VmlldzoxNjg0Njk1MA/edit?draftId=VmlldzoxNjg0Njk1MA==

# github link - https://github.com/Aravindhan-Mohanraj/DA6401_Assignment_3.git

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
