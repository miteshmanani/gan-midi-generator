# MIDI Music Generation with Conditional WGAN-GP

This repository contains a Jupyter Notebook implementation of a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) for generating MIDI piano rolls conditioned on musical keys (e.g., C major, D minor). The model generates piano roll representations of music (128 notes × 32 time steps) using PyTorch, with key conditioning to control the musical output.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Project Overview
This project implements a conditional WGAN-GP to generate MIDI music. The Generator produces piano rolls conditioned on one of 24 musical keys (12 major, 12 minor), while the Discriminator evaluates the authenticity of generated samples. The model uses Wasserstein loss with gradient penalty for stable training and PrettyMIDI for MIDI processing.

The code is written in a Jupyter Notebook (`train.ipynb`) and includes modular components for the Generator, Discriminator, and utility functions for MIDI processing.

## Features
- **Conditional Generation**: Generate music in specific musical keys (e.g., C major, D minor).
- **Stable Training**: Uses WGAN-GP with gradient penalty to avoid mode collapse.
- **MIDI Processing**: Converts MIDI files to piano rolls and estimates musical keys using chroma analysis.
- **Evaluation Metrics**: Computes note density and pitch entropy for generated piano rolls.
- **Output**: Saves generated MIDI files every 50 epochs and for selected keys at the end.

## Requirements
- Python version: 3.10.9 (tags/v3.10.9:1dd9be6, Dec  6 2022, 20:01:21) [MSC v.1934 64 bit (AMD64)]
- NumPy version: 1.26.4
- PyTorch version: 2.2.2+cu118
- pretty_midi version: 0.2.10
- CUDA version: 11.8
- GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/gan-midi-generator.git
   cd gan-midi-generator

Create a virtual environment (optional but recommended):
bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bash

pip install torch numpy pretty_midi

Prepare the dataset:
Place MIDI files (.mid or .midi) in a data folder in the project root.

The code processes up to 1000 MIDI files by default.

Usage
Open the Jupyter Notebook:
bash

jupyter notebook MidiGAN.ipynb

Configure the notebook:
Ensure the data folder contains MIDI files.

Adjust hyperparameters (e.g., latent_dim, batch_size, epochs) in the notebook if needed.

Run the notebook:
Execute all cells to train the model.

The model saves generated MIDI files in the output folder every 50 epochs and final outputs for selected keys.

View outputs:
Check the output folder for MIDI files (e.g., generated_epoch_0050_cmajor.mid).

Use a MIDI player (e.g., VLC, MuseScore) to listen to the generated music.

File Structure

gan-midi-generator/
├── MidiGAN.ipynb              # Main Jupyter Notebook with training loop
├── data/                   # Folder for MIDI files (not included)
├── output/                 # Folder for generated MIDI files and model checkpoints
└── README.md               # This file

Dataset
Source: MIDI files (.mid or .midi) in the data folder.

Processing:
Converted to piano rolls (128 notes × 32 time steps) using PrettyMIDI.

Musical keys estimated via chroma analysis (24 keys: 12 major, 12 minor).

Requirements:
Place MIDI files in the data folder.

Ensure files are valid and contain piano-based music for best results.

Training
Model: Conditional WGAN-GP with Generator and Discriminator.

Hyperparameters:
Latent dimension: 256

Batch size: 64

Epochs: 5000

Learning rate: 0.0001

Gradient penalty weight: 10

Optimizer: Adam (β1=0.5, β2=0.9)

Training Process:
Discriminator trained 5 times per epoch (n_critic=5).

Generator trained to minimize Wasserstein loss.

Outputs saved every 50 epochs in output/.

Evaluation
Metrics:
Note Density: Measures sparsity of generated piano rolls.

Pitch Entropy: Assesses diversity of pitch usage.

Qualitative:
Listen to generated MIDI files to evaluate musical coherence.

Outputs conditioned on keys (e.g., C major, D minor) for comparison.

Contributing
Contributions are welcome! Please:
Fork the repository.

Create a feature branch (git checkout -b feature/new-feature).

Commit changes (git commit -m "Add new feature").

Push to the branch (git push origin feature/new-feature).

Open a pull request.
