# [IEEE Access] Contrastive Disentangled Variational Autoencoder for Collaborative Filtering.
[![View Paper](https://img.shields.io/badge/View%20Paper-IEEE-00629B)](https://ieeexplore.ieee.org/document/11023251) [![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2025.3576445-blue)](https://doi.org/10.1109/ACCESS.2025.3576445)

## 📄 Paper 

This repository contains the implementation code for the CD-VAE and ECD-VAE models as proposed in our paper.

## Requirements
The required libraries and packages are listed in ```requirements.txt```.

## Datasets
We used the MovieLens20M and Netflix datasets in our experiments. 

```./datasets``` include download paths and preprocessing procedures.

## Example Usage

Here's an example of how to train the CD-VAE model on the MovieLens20M dataset:

```bash
python experiment.py --max_beta=0.3 --gated --input_type="binary" --s1_size=200 --s2_size=200 --z_size=200 --hidden_size=600 --num_layers=2 --note="ml20m(CD-VAE)"
```

## Acknowledgements
Our code is based on https://github.com/dawenl/vae_cf and https://github.com/psywaves/EVCF .
