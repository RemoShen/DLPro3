# Jailbreaking Deep Models: Adversarial Attacks on ImageNet Classifiers

## Project Overview

This project aims to systematically explore and implement adversarial attacks on public deep image classifiers, focusing on ResNet-34 trained on ImageNet-1K. We implement and evaluate various attack methods (FGSM, iterative attacks, patch attacks, etc.) on a subset of ImageNet-1K, and analyze their effectiveness and transferability.

## Directory Structure

```
├── Remo_modify_Complete.ipynb   # Main Jupyter Notebook with code and experiments
├── TestDataSet/                 # Test image dataset (100 classes, 500 images)
├── Adversarial_Test_Set_1/      # Adversarial samples from FGSM attack
├── Adversarial_Test_Set_2/      # Adversarial samples from improved attack
├── Adversarial_Test_Set_3/      # Adversarial samples from patch attack
└── README.md                    # Project documentation
```

## Requirements

- Python 3.8+
- torch >= 1.10
- torchvision >= 0.11
- numpy
- matplotlib
- tqdm
- jupyter

## How to Run

1. Download and extract `TestDataSet` into the project root directory.
2. Open `Remo_modify_Complete.ipynb` and run all cells in order to reproduce the experiments and visualizations.
3. Generated adversarial samples will be saved in `Adversarial_Test_Set_1/2/3` folders automatically.

## Main Tasks

1. **Baseline Evaluation**  
   - Load the pre-trained ResNet-34 model from torchvision (weights='IMAGENET1K_V1').
   - Preprocess images using standard ImageNet normalization.
   - Evaluate the model on the test set and report top-1 and top-5 accuracy.
2. **Pixel-wise Attack (FGSM)**  
   - Implement the Fast Gradient Sign Method (FGSM) with ε=0.02.
   - For each image, compute the gradient of the cross-entropy loss with respect to the input, take the sign, and perturb the image accordingly.
   - Visualize several adversarial examples and verify the L∞ constraint.
   - Save the adversarial images as "Adversarial Test Set 1" and evaluate the model's accuracy on this set.
3. **Improved Attacks**  
   - Implement the Projected Gradient Descent (PGD) attack with multiple steps and random starts, under the same ε=0.02 constraint.
   - Optionally explore targeted attacks and other optimizers.
   - Visualize and save the new adversarial images as "Adversarial Test Set 2" and evaluate the model's accuracy.
4. **Patch Attacks**  
   - Apply the best performing attack (PGD) but restrict perturbations to a random 32x32 patch in each image.
   - To improve attack effectiveness, we use Grad-CAM to generate a saliency map for each image, identifying the most important region for the model's prediction.
   - The 32x32 patch is then placed at the location with the highest Grad-CAM activation, focusing the perturbation on the most influential area.
   - Allow a larger ε (0.5) for the patch region to maximize the attack effect within the limited area.
   - Visualize and save the adversarial images as "Adversarial Test Set 3" and evaluate the model's accuracy.
5. **Transferability Evaluation**  
   - Evaluate all four datasets (original + 3 adversarial sets) on another pre-trained model (e.g., DenseNet-121 from torchvision).
   - Report and compare top-1/top-5 accuracy for each dataset, and analyze the transferability of the attacks.

## Results Reproduction

- All experimental results (top-1/top-5 accuracy, visualizations, perturbation sizes, etc.) are clearly output in the notebook.
- You can directly run the notebook or call the relevant functions for batch evaluation.


## Team Members

- Jialin Li
- Mike Poon
- Remo Shen