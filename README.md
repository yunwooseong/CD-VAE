# [IEEE Access] Contrastive Disentangled Variational Autoencoder for Collaborative Filtering.
[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-E24D35)](https://ieeexplore.ieee.org/document/11023251) [![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2025.3576445-blue)](https://doi.org/10.1109/ACCESS.2025.3576445)

## 📄 Paper 

This repository contains the implementation code for the CD-VAE and ECD-VAE models as proposed in our paper.

## 📊 Datasets
We used the MovieLens20M and Netflix datasets in our experiments. 

```./datasets``` include download paths and preprocessing procedures.

## 🛠️ Requirements
The required libraries and packages are listed in ```requirements.txt```.

## 🚀 Run

Here's an example of how to train the CD-VAE model on the MovieLens20M dataset:

```bash
python experiment.py --max_beta=0.3 --gated --input_type="binary" --s1_size=200 --s2_size=200 --z_size=200 --hidden_size=600 --num_layers=2 --note="ml20m(CD-VAE)"
```

## 🙏 Acknowledgements
This work was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00419201) and Institute of Information \& Communications Technology Planning \& Evaluation (IITP) grant funded by the Korean government (MSIT) (RS-2021-II211341, Artificial Intelligence Graduate School Program of Chung-Ang Univ.). This research was supported by the Chung-Ang University Graduate Research Scholarship in 2023.
