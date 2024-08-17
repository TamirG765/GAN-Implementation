import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving checkpoints")
    parser.add_argument("--load_checkpoint", action='store_true', help="path to the checkpoint file to load")
    opt = parser.parse_args()
    print("Args: ", opt)

    # Check CUDA / MPS availability and set the device accordingly
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"MPS device is available: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA device is available: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA/MPS device is not available. Using CPU.")

    # Tensor type
    if device == "mps" and torch.backends.mps.is_available():
        Tensor = torch.mps.FloatTensor
    elif device == "cuda" and torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    # PATH for dataset
    PATH = '/Users/tamir_gez/Documents/Study/סמסטר ב שנה ד/Machine Learning/ML Project/celebA'

    def compute_fid(generator, real_images_path, temp_folder, device):
        # Ensure generator is in eval mode
        generator.eval()
        
        # Generate images and save to temporary directory
        with torch.no_grad():
            z = torch.randn(opt.batch_size, opt.latent_dim, device=device)
            fake_images = generator(z)
            fake_images = (fake_images + 1) / 2  # Rescale images to [0, 1]
            for i, image in enumerate(fake_images):
                save_path = os.path.join(temp_folder, f'image_{i:04d}.png')
                vutils.save_image(image, save_path, normalize=True)
        
        # Calculate FID score
        fid = calculate_fid_given_paths([real_images_path, temp_folder], batch_size=opt.batch_size, device=device, dims=2048)
        
        # Switch back to train mode
        generator.train()
        return fid

    # Image shape for the network
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(opt.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *img_shape)
            return img

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Configure data loader for CelebA dataset
    dataloader = DataLoader(
        datasets.ImageFolder(
            root=PATH,
            transform=transforms.Compose([
                transforms.Resize((opt.img_size, opt.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load checkpoint if specified
    if opt.load_checkpoint:
        checkpoint = torch.load('checkpoints/ckpt_epoch7.pth')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        d_loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with discriminator loss: {d_loss}")
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, opt.n_epochs):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit='batch') as pbar:
            for i, (imgs, _) in enumerate(dataloader):
                
                # Adversarial ground truths
                valid = torch.ones((imgs.size(0), 1), device=device, requires_grad=False)  # Real samples label
                fake = torch.zeros((imgs.size(0), 1), device=device, requires_grad=False)  # Fake samples label
                z = torch.randn((imgs.size(0), opt.latent_dim), device=device)  # Noise vector for generator
                
                # Configure input
                real_imgs = Variable(imgs.type(Tensor)).to(device)
                
                # Train Generator
                optimizer_G.zero_grad()
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                pbar.update(1)
                pbar.set_postfix({'D loss': d_loss.item(), 'G loss': g_loss.item()})

            # Compute and display FID score at the end of each epoch
            vutils.save_image(gen_imgs.data[:25], f"output/{epoch}_.png", nrow=5, normalize=True)
            temp_folder = 'temp_gen_images'
            os.makedirs(temp_folder, exist_ok=True)  # Ensure the directory exists
            real_images_path = 'celebA/img_align_celeba/img_align_celeba' 
            fid_value = compute_fid(generator, real_images_path, temp_folder, device)
            print(f'Epoch {epoch + 1}: FID Score = {fid_value}')
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,  # plus one because epochs are zero-indexed
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss': d_loss.item(),  # Saving the last discriminator loss
            }
            torch.save(checkpoint, f"checkpoints/ckpt_epoch{epoch+1}.pth")
            
if __name__ == '__main__':
    main()