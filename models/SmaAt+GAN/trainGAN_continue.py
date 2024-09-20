from data_process.data_loader import PlantImage_Dataset
from torch.utils.data import DataLoader
import torch
from models.SmaAt_UNet import SmaAt_UNet
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks import Discriminator

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias.data, 0)

def train_model():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Parameter settings
    path_plant_images = 'plant_generate/'
    path_Model_Save_model_paras_ori = 'best_generator.pth'
    path_discriminator_paras_ori = 'best_discriminator.pth'
    n_channels_ori = 5
    n_classes = 20
    p_valid = 1
    num_workers = 4  # Increase data loading efficiency
    lr_G = 0.0001  # Generator learning rate
    lr_D = 0.0004  # Discriminator learning rate
    batch_size = 24  # Increase batch size

    # Initialize generator and discriminator
    generator = SmaAt_UNet(n_channels=n_channels_ori, n_classes=n_classes).to(device)
    generator.load_state_dict(torch.load(path_Model_Save_model_paras_ori, map_location=lambda storage, loc: storage))
    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load(path_discriminator_paras_ori, map_location=lambda storage, loc: storage))

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Define loss functions and optimizers
    criterion = torch.nn.BCELoss().to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # Use schedulers to adjust learning rates dynamically
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)

    # Load data
    plant_image_dataset_train = PlantImage_Dataset('train', path_data=path_plant_images)
    dataloader_train = DataLoader(plant_image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # shuffle=True
    plant_image_dataset_valid = PlantImage_Dataset('valid', path_data=path_plant_images)
    dataloader_valid = DataLoader(plant_image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Define training parameters
    num_epochs = 100
    real_label = 1
    fake_label = 0
    patience = 10  # Reduce patience value to accelerate early stopping

    # Create a txt file to save results
    with open('loss_results.txt', 'w') as f:
        f.write('Weight\tBCE_Loss\tMSE_Loss\n')

    # Record the best BCE Loss and corresponding models
    best_global_bce_loss = float('inf')
    best_global_generator_state = None
    best_global_discriminator_state = None

    # Training
    print("Starting Training Loop...")

    for weight in np.arange(0.01, 0.51, 0.01):
        best_val_loss = float('inf')
        best_val_bce_loss = float('inf')
        best_generator_state = None
        best_discriminator_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            generator.train()
            discriminator.train()
            i = 0
            for inputs, targets in dataloader_train:
                # Update discriminator
                inputs = inputs.to(device)
                targets = targets.to(device)
                discriminator.zero_grad()
                b_size = targets.size(0)
                label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
                
                output_real = discriminator(targets).view(-1)
                errD_real = criterion(output_real, label_real)

                fake = generator(inputs)
                output_fake = discriminator(fake.detach()).view(-1)
                errD_fake = criterion(output_fake, label_fake)

                errD = errD_real + errD_fake
                errD.backward()
                optimizer_D.step()

                D_x = output_real.mean().item()
                D_G_z1 = output_fake.mean().item()

                # Update generator
                generator.zero_grad()
                output = discriminator(fake).view(-1)
                errG = (weight) * criterion(output, label_real) + (1 - weight) * mse_loss(fake, targets)
                errG.backward()
                optimizer_G.step()

                D_G_z2 = output.mean().item()

                i += 1

            # Validation phase
            generator.eval()
            val_bce_loss = 0.0
            val_mse_loss = 0.0
            with torch.no_grad():
                for inputs, targets in dataloader_valid:
                    b_size = targets.size(0)
                    label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    fake = generator(inputs)
                    output = discriminator(fake).view(-1)
                    val_bce_loss += criterion(output, label_real).item()
                    val_mse_loss += mse_loss(fake, targets).item()

            val_bce_loss /= len(dataloader_valid)
            val_mse_loss /= len(dataloader_valid)
            val_loss = (1 - weight) * val_bce_loss + (weight) * val_mse_loss

            # Save the best model and loss values
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_bce_loss = val_bce_loss
                best_generator_state = generator.state_dict()
                best_discriminator_state = discriminator.state_dict()
                # Create folder
                weight_folder = f'weight_{weight:.2f}'
                os.makedirs(weight_folder, exist_ok=True)
                torch.save(best_generator_state, os.path.join(weight_folder, 'best_generator.pth'))
                torch.save(best_discriminator_state, os.path.join(weight_folder, 'best_discriminator.pth'))
                epochs_no_improve = 0
                print(f'Current weight: {weight:.2f} - Validation BCE Loss: {val_bce_loss:.4f} MSE Loss: {val_mse_loss:.4f} Weighted Loss: {val_loss:.4f}')
                with open('loss_results.txt', 'a') as f:
                    f.write(f'{weight:.2f}\t{val_bce_loss:.4f}\t{val_mse_loss:.4f}\n')
            else:
                epochs_no_improve += 1

            # Early stopping mechanism
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1} for weight {weight:.2f}')
                break

            # Update global best model
            if best_val_bce_loss < best_global_bce_loss:
                best_global_bce_loss = best_val_bce_loss
                best_global_generator_state = best_generator_state
                best_global_discriminator_state = best_discriminator_state

        # Adjust learning rates
        scheduler_G.step()
        scheduler_D.step()

    # Save the final best global models
    if best_global_generator_state and best_global_discriminator_state:
        torch.save(best_global_generator_state, 'best_generator_global.pth')
        torch.save(best_global_discriminator_state, 'best_discriminator_global.pth')

    print(f'The weight with the lowest validation BCE loss is: {best_global_bce_loss}')

if __name__ == "__main__":
    train_model()
