import torch
import torch.optim as optim
import numpy as np
from model import ConfGAN, energy_loss

############################################
#  Hyperparameters (as in Nature Methods)
############################################
BATCH_SIZE = 64
Z_DIM = 128
EPOCHS = 1000
LR = 2e-4
N_ATOMS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
#  Load training data
#  (Distances + motif embeddings from DFT-optimized structures)
############################################

def load_dataset():
    """
    Each sample:
    - motif_tensor: (n_motifs, node_dim)
    - D_real: (n_atoms, n_atoms)
    - E_real: scalar (DFT energy)
    """
    data = np.load("../data/confgan_training.npz")
    motifs = torch.tensor(data["motifs"]).float()
    distances = torch.tensor(data["distances"]).float()
    energies = torch.tensor(data["energies"]).float()
    return motifs, distances, energies


motifs, D_real, E_real = load_dataset()

############################################
#  Initialize ConfGAN
############################################

model = ConfGAN(n_atoms=N_ATOMS).to(DEVICE)
G = model.G
D = model.D
MMG = model.mmg

optG = optim.Adam(list(G.parameters()) + list(MMG.parameters()), lr=LR, betas=(0.5,0.999))
optD = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))

bce = torch.nn.BCELoss()

############################################
#  Training Loop
############################################

for epoch in range(EPOCHS):
    perm = torch.randperm(len(motifs))

    for i in range(0, len(motifs), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]

        motif_batch = motifs[idx].to(DEVICE)
        D_batch = D_real[idx].to(DEVICE)
        E_batch = E_real[idx].to(DEVICE)

        ####################################
        #  Train Discriminator
        ####################################
        cond = MMG(motif_batch)
        z = torch.randn(len(idx), Z_DIM).to(DEVICE)
        D_fake, E_fake = G(z, cond)

        real_labels = torch.ones(len(idx),1).to(DEVICE)
        fake_labels = torch.zeros(len(idx),1).to(DEVICE)

        pred_real = D(D_batch, E_batch)
        pred_fake = D(D_fake.detach(), E_fake.detach())

        lossD = bce(pred_real, real_labels) + bce(pred_fake, fake_labels)

        optD.zero_grad()
        lossD.backward()
        optD.step()

        ####################################
        #  Train Generator + MM-GNN
        ####################################
        pred_fake = D(D_fake, E_fake)
        gan_loss = bce(pred_fake, real_labels)

        phys_loss = energy_loss(D_fake)

        lossG = gan_loss + 0.1 * phys_loss

        optG.zero_grad()
        lossG.backward()
        optG.step()

    ####################################
    #  Logging (Nature-quality)
    ####################################
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | LossD={lossD.item():.4f} | LossG={lossG.item():.4f}")

        torch.save(model.state_dict(), f"confgan_epoch_{epoch}.pt")

print("ConfGAN training completed.")
