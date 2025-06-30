import torch
import torch.nn.functional as F
import numpy as np
import random

def train_model(model, classifier, features_tensor, times_tensor, time_embed, noise_scheduler, optimizer, num_epochs, guidance_weight=1.0):
    criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        losses = []
        for i in range(len(features_tensor)):
            t = random.randint(0, noise_scheduler.config.num_train_timesteps - 1)
            timesteps = torch.tensor([t], dtype=torch.long).to(features_tensor.device)

            noise = torch.randn_like(features_tensor[i].unsqueeze(0))
            noisy = noise_scheduler.add_noise(features_tensor[i].unsqueeze(0), noise, timesteps)
            time_encoding = time_embed(times_tensor[i].unsqueeze(0))

            pred = model(noisy, time_encoding)
            diffusion_loss = F.mse_loss(pred, noise)

            if i > 0 and i < len(features_tensor) - 1:
                random_index = i + 1 if random.random() < 0.5 else i - 1
            elif i == 0:
                random_index = i + 1
            else:
                random_index = i - 1

            combined = torch.cat([features_tensor[i].unsqueeze(0), features_tensor[random_index].unsqueeze(0)], dim=1)
            order_label = torch.tensor([0 if random_index > i else 1], dtype=torch.long).to(features_tensor.device)
            order_pred = classifier(combined)
            classifier_loss = criterion(order_pred, order_label)

            total_loss = diffusion_loss + guidance_weight * classifier_loss
            losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    return epoch_losses

@torch.no_grad()
def validate_model(model, classifier, features_tensor, times_tensor, time_embed, noise_scheduler, guidance_weight=1.0):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    classifier.eval()
    losses = []
    for i in range(len(features_tensor)):
        t = random.randint(0, noise_scheduler.config.num_train_timesteps - 1)
        timesteps = torch.tensor([t], dtype=torch.long).to(features_tensor.device)

        noise = torch.randn_like(features_tensor[i].unsqueeze(0))
        noisy = noise_scheduler.add_noise(features_tensor[i].unsqueeze(0), noise, timesteps)
        time_encoding = time_embed(times_tensor[i].unsqueeze(0))

        pred = model(noisy, time_encoding)
        diffusion_loss = F.mse_loss(pred, noise)

        if i > 0 and i < len(features_tensor) - 1:
            random_index = i + 1 if random.random() < 0.5 else i - 1
        elif i == 0:
            random_index = i + 1
        else:
            random_index = i - 1

        combined = torch.cat([features_tensor[i].unsqueeze(0), features_tensor[random_index].unsqueeze(0)], dim=1)
        order_label = torch.tensor([0 if random_index > i else 1], dtype=torch.long).to(features_tensor.device)
        order_pred = classifier(combined)
        classifier_loss = criterion(order_pred, order_label)

        total_loss = diffusion_loss + guidance_weight * classifier_loss
        losses.append(total_loss.item())
    return np.mean(losses)
