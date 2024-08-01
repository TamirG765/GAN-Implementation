# GAN-Implementation

## Intro
Generative Adversarial Networks (GANs) are a class of machine learning frameworks.
<br>GANs consist of two neural networks, the generator and the discriminator, which compete against each other.
<br>The generator creates fake data, while the discriminator attempts to distinguish between real and fake data. This adversarial process continues until the generator produces data that is indistinguishable from real data.

## How to Run

### Step 1: Create and Activate a New Conda Environment
1. **Create a new conda environment with Python 3.9.13**:
<br>`conda create --name gan-env python=3.9.13`
2. **Activate the new environment:**
<br>`conda activate gan-env`

### Step 2: Install Dependencies:
<br>`pip install -r requirements.txt`

### Step 3: Run the Code:
<br>`python GAN.py