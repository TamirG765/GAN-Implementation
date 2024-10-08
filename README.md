# GAN-Implementation 

## Intro - 🤖 VS 🤖
Generative Adversarial Networks (GANs) are a type of machine learning model.
<br>They involve two neural networks: the generator and the discriminator. 
<br>These networks compete with each other, with the generator creating synthetic data and the discriminator trying to identify whether the data is real or fake.
<br>This adversarial process helps the generator improve until it produces data that is nearly indistinguishable from real data.

## How to Run the Project

### Step 1: Create and Activate a New Conda Environment

#### 1. Create a new conda environment with Python 3.9.13:
`conda create --name gan-env python=3.9.13`
#### 2. Activate the new environment:
`conda activate gan-env`

### Step 2: Install Dependencies:
`pip install -r requirements.txt`

### Step 3: Download the Training Set
The training set is very large and can be downloaded from the following link:
[Download CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

### Step 4: Run the Code:
`python GAN.py`