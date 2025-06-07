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
- Python 3.8+
- PyTorch
- NumPy
- PrettyMIDI
- A dataset of MIDI files (not included in this repository)

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

License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Instructions for Use
1. **Copy the Text**: Copy the entire block above.
2. **Create README.md**:
   - In your GitHub repository, create a file named `README.md`.
   - Paste the copied text into `README.md`.
3. **Update Placeholder**:
   - Replace `your-username` in the `git clone` command with your actual GitHub username.
4. **Optional Customizations**:
   - If you have a specific dataset (e.g., MAESTRO, Lakh MIDI Dataset), add a link or description in the "Dataset" section.
   - If you want to include sample MIDI outputs or a demo link, add a section like `## Sample Outputs` with links to files or a demo page.
   - If you have a `LICENSE` file, ensure it exists in the repository root, or remove the link if you’re not including one.
5. **Commit and Push**:
   ```bash
   git add README.md
   git commit -m "Add README file"
   git push origin main

Notes
The README assumes the repository name is gan-midi-generator. Update the clone URL if your repository has a different name.

The file structure section matches the code you provided. If your .ipynb file has a different name or if you add more files, update the File Structure section.

If you want to add badges (e.g., for Python version or license), let me know, and I can include them.


Let me know if you need further tweaks, such as adding badges, specific dataset instructions, or a different tone for the README!

explain MIDI processing

other GAN variants

more concise instructions

