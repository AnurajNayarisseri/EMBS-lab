import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z):
        return self.model(z)


# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# GAN Training
# -----------------------------
def train_gan(generator, discriminator, real_latents,
              epochs=10, batch_size=32, lr=1e-4, device="cpu"):

    generator.to(device)
    discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    criterion = nn.BCELoss()

    real_latents = torch.tensor(real_latents, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        perm = torch.randperm(real_latents.size(0))
        total_d_loss = 0
        total_g_loss = 0

        for i in range(0, real_latents.size(0), batch_size):
            idx = perm[i:i + batch_size]
            real = real_latents[idx]

            batch_size_current = real.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            noise = torch.randn(batch_size_current, real.size(1)).to(device)
            fake = generator(noise)

            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)

            optimizer_D.zero_grad()

            real_loss = criterion(discriminator(real), real_labels)
            fake_loss = criterion(discriminator(fake.detach()), fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            fake = generator(noise)
            g_loss = criterion(discriminator(fake), real_labels)

            g_loss.backward()
            optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        print(f"Epoch {epoch+1}/{epochs} | D Loss: {total_d_loss:.4f} | G Loss: {total_g_loss:.4f}")

    return generator, discriminator


# -----------------------------
# Sampling Latent Vectors
# -----------------------------
def generate_latents(generator, num_samples, latent_dim=64, device="cpu"):
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_latents = generator(noise)

    return fake_latents.cpu().numpy()


# -----------------------------
# CONDITIONAL GAN (Optional)
# -----------------------------
class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=10, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.model(x)


class CGAN_Discriminator(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=10, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)


# -----------------------------
# Debug / Example Run
# -----------------------------
if __name__ == "__main__":
    import numpy as np

    # Fake latent dataset (simulate VAE output)
    real_latents = np.random.randn(500, 64)

    generator = Generator(latent_dim=64)
    discriminator = Discriminator(latent_dim=64)

    print("Training GAN...")
    generator, discriminator = train_gan(generator, discriminator, real_latents, epochs=3)

    print("\nGenerating new latent vectors...")
    new_latents = generate_latents(generator, num_samples=5)

    print("Generated Latents:")
    print(new_latents)