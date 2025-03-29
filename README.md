# STFT-KAN Framework

## Overview

We propose a new **Kolmogorov-Arnold Network (KAN)**, called **STFT-KAN**, which introduces **Short-Time Fourier Transform (STFT)** into KAN. The **STFT-KAN framework** allows for the use of different window sizes across various frequency ranges, providing a more adaptable and efficient way to represent underlying signals by capturing nonstationary frequency characteristics. This enhanced Fourier-based KAN formulation offers better control over model parameters, reducing the risk of overfitting while preserving performance similar to that of the original KAN model.

<img src="media/STFT-KAN.jpg" alt="STFT-KAN" width="70%">

We have two sections: the first section [STFT-KAN Benchmarking](#stft-kan-benchmarking) focuses on evaluating STFT-KAN on general tasks like MNIST digit classification. The second section [Paper Implementation Code](#paper-implementation-code) details the method introduced in our paper, where STFT-KAN is applied to 3D point cloud classification of tree species using Dynamic Graph CNN (DGCNN).

---

## STFT-KAN Benchmarking

### STFT-KAN on MNIST

To experiment with **STFT-KAN** on a general task like **MNIST digit classification**, execute:
```bash
cd STFT-KAN-MNIST  
python stft-kan-mnist.py
```
## Paper implementation code

### How to Execute

1. **Download Dataset:**
   - We already have a **preprocessed dataset**, it is stored in the `data` folder in **h5 format**.
   - Else, you can preprocess the dataset **STPCTLS** used in this study, which is publicly available and can be accessed via the following link: [STPCTLS Dataset](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/FOHUJM).

2. **Set up the environment:**
   - This project uses **PyTorch 2.4.1** with **CUDA 11.8**. Ensure you have the required dependencies by installing from the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

3. **Grant Execution Access:**
   - Before training, ensure the training script has execution permissions:
     ```bash
     chmod +x train.sh
     ```

4. **Train the Models:**
   - To start training, simply execute the following script:
     ```bash
     ./train.sh
     ```
  ### Bayesian Optimization of Hyperparameters for STFT-KAN

The code for Bayesian optimization is in the *hyperparameter tuning* folder. You can simply run the following command:

```bash
python bayesian_optimization.py
```
### Citation
If you use this code in your research, please cite:
```bibtex
@article{ohamouddou2025STFKAN,
  author    = {Said Ohamouddou and Mohamed Ohamouddou and Rafik Lasri and Hanaa El Afia and Raddouane Chiheb and Abdellatif El Afia},
  title     = {Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic Graph CNN Approach for Tree Species Classification in 3D Point Clouds},
  year      = {2025},
  month     = {March 29},
  note      = {Available at SSRN: \url{https://ssrn.com/abstract=}},
}
