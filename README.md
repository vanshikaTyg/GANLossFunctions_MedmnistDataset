# GANLossFunctions_MedmnistDataset
# GAN Loss Function Comparison

## Overview
This project explores the impact of different loss functions on GAN performance using the MedMNIST dataset. Three GAN variants are implemented and evaluated:

- **Least Squares GAN (LS-GAN)**
- **Wasserstein GAN (WGAN)**
- **WGAN with Gradient Penalty (WGAN-GP)**

Each model is trained for at least 50 epochs, and performance is measured using:

- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**
- **Visual Inspection**

TensorBoard is used for visualizing generated images and training progress.

---

## Dataset
The MedMNIST dataset is used for training. It is a collection of lightweight medical image datasets. More details can be found [here](https://medmnist.com/).

---

## Installation
Ensure you have the necessary dependencies installed:

```sh
pip install torch torchvision torchmetrics torch-fidelity tensorboard tqdm numpy medmnist Pillow
```

---

## Running the Code
### Clone the repository
```sh
git clone https://github.com/vanshikaTyg/GANLossFunctions_MedmnistDataset
cd <repo-folder>
```

### Train the GAN models
```sh
python train.py --model ls_gan
python train.py --model wgan
python train.py --model wgan_gp
```

### Evaluate the models
```sh
python evaluate.py
```

### Visualize results with TensorBoard
```sh
tensorboard --logdir=runs
```

---

## Results and Evaluation
### Model Performance
| GAN Model  | FID Score  | Inception Score |
|------------|------------|----------------|
| LS-GAN     | 387.7719   | 1.0000         |
| WGAN       | 334.1937   | 1.0000         |
| WGAN-GP    | 332.1943   | 1.0000         |

- Generated images are stored in the `outputs/` directory.
- TensorBoard logs are saved under `runs/`.
- IS and FID scores are computed for model comparison.

### Conclusion
Based on the FID scores, WGAN-GP performs the best among the three models, achieving the lowest FID score (332.1943), indicating better quality image generation. WGAN also performs well with a slightly higher FID score (334.1937). LS-GAN has the highest FID score (387.7719), suggesting it generates less realistic images. All models yield the same Inception Score of 1.0000, which may indicate the need for further evaluation or dataset adjustments.

---

## Authors
Vanshika Tyagi

