import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoder
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    # -----------------------------
    # Encoder
    # -----------------------------
    def encode(self, x):
        embedded = self.embedding(x)
        _, (h, _) = self.encoder_lstm(embedded)
        h = h[-1]

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    # -----------------------------
    # Reparameterization
    # -----------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -----------------------------
    # Decoder
    # -----------------------------
    def decode(self, z, seq_len):
        h = self.decoder_input(z)
        h = h.unsqueeze(1).repeat(1, seq_len, 1)

        out, _ = self.decoder_lstm(h)
        logits = self.output_layer(out)

        return logits

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        recon_logits = self.decode(z, x.size(1))

        return recon_logits, mu, logvar


# -----------------------------
# Loss Function
# -----------------------------
def vae_loss(recon_logits, target, mu, logvar):
    # Reconstruction loss (CrossEntropy)
    recon_loss = nn.functional.cross_entropy(
        recon_logits.view(-1, recon_logits.size(-1)),
        target.view(-1),
        ignore_index=0  # <pad>
    )

    # KL Divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss


# -----------------------------
# Training Function
# -----------------------------
def train_vae(model, data, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data = torch.tensor(data, dtype=torch.long)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        perm = torch.randperm(data.size(0))

        for i in range(0, data.size(0), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = data[batch_idx].to(device)

            optimizer.zero_grad()

            recon_logits, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(recon_logits, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model


# -----------------------------
# Sampling (Generate Molecules)
# -----------------------------
def sample(model, num_samples, seq_len, device="cpu"):
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        logits = model.decode(z, seq_len)

        probs = torch.softmax(logits, dim=-1)
        tokens = torch.argmax(probs, dim=-1)

    return tokens.cpu().numpy()


# -----------------------------
# Decode Tokens → SELFIES
# -----------------------------
def tokens_to_selfies(token_batch, itos):
    selfies_list = []

    for seq in token_batch:
        tokens = [itos.get(int(i), "") for i in seq]

        # Remove special tokens
        tokens = [t for t in tokens if t not in ["<pad>", "<start>", "<end>"]]

        selfies = "".join(tokens)
        selfies_list.append(selfies)

    return selfies_list


# -----------------------------
# Full Generation Pipeline
# -----------------------------
def generate_molecules(model, num_samples, seq_len, itos, device="cpu"):
    token_batch = sample(model, num_samples, seq_len, device)
    selfies_list = tokens_to_selfies(token_batch, itos)

    return selfies_list


# -----------------------------
# Debug Run
# -----------------------------
if __name__ == "__main__":
    import numpy as np

    # Fake dataset (for testing)
    vocab_size = 20
    data = np.random.randint(0, vocab_size, (100, 50))

    model = VAE(vocab_size)

    print("Training VAE...")
    model = train_vae(model, data, epochs=2)

    print("Generating samples...")
    samples = sample(model, num_samples=5, seq_len=50)

    print("Sample tokens:")
    print(samples)