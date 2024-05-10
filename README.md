# Contrastive Disentangled Variational Autoencoder for Collaborative Filtering.

### CD-VAE and ECD-VAE Implementation

This repository contains the implementation code for the CD-VAE and ECD-VAE models as proposed in our paper.

### Example Usage

Here's an example of how to train the CD-VAE model on the MovieLens20M dataset:

```bash
python experiment.py --max_beta=0.3 --gated --input_type="binary" --s1_size=200 --s2_size=200 --z_size=200 --hidden_size=600 --num_layers=2 --note="ml20m(CD-VAE)"
```
### Requirements
The required libraries and packages are listed in ```requirements.txt```.
### Datasets
We used the MovieLens20M and Netflix Prize datasets in our experiments. 

```./datasets``` include download paths and preprocessing procedures.

### Acknowledgements
Our code is based on https://github.com/dawenl/vae_cf and https://github.com/psywaves/EVCF .
