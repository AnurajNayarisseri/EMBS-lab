import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from vae import MolecularVAE, vae_loss
from gan import MolGenerator, MolDiscriminator
from rl import reward

############################################
# Hyperparameters (Nature Methods)
############################################

FP_DIM = 2048
LATENT = 128
BATCH = 64
EPOCHS = 300
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Load CDK2 training molecules
############################################

df = pd.read_csv("../data/cdk2_inhibitors.csv")   # SMILES column
smiles = df["SMILES"].tolist()

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=FP_DIM)
    return np.array(fp)

X = np.array([featurize(s) for s in smiles])
X = torch.tensor(X).float().to(DEVICE)

############################################
# Models
############################################

vae = MolecularVAE(FP_DIM, LATENT).to(DEVICE)
G = MolGenerator(LATENT, FP_DIM).to(DEVICE)
D = MolDiscriminator(FP_DIM).to(DEVICE)

optVAE = optim.Adam(vae.parameters(), LR)
optG = optim.Adam(G.parameters(), LR)
optD = optim.Adam(D.parameters(), LR)

bce = torch.nn.BCELoss()

############################################
# Training loop
############################################

for epoch in range(EPOCHS):
    perm = torch.randperm(len(X))

    for i in range(0,len(X),BATCH):
        idx = perm[i:i+BATCH]
        x = X[idx]

        ######################################
        # Train VAE
        ######################################

        recon, mu, logvar = vae(x)
        loss_vae = vae_loss(x, recon, mu, logvar)

        optVAE.zero_grad()
        loss_vae.backward()
        optVAE.step()

        ######################################
        # GAN
        ######################################

        z = torch.randn(len(idx), LATENT).to(DEVICE)
        fake_fp = G(z)
        real_pred = D(x)
        fake_pred = D(fake_fp.detach())

        lossD = bce(real_pred, torch.ones_like(real_pred)) + \
                bce(fake_pred, torch.zeros_like(fake_pred))

        optD.zero_grad()
        lossD.backward()
        optD.step()

        ######################################
        # RL-guided Generator
        ######################################

        fake_pred = D(fake_fp)
        smiles_batch = smiles[:len(idx)]
        rewards = torch.tensor([reward(s) for s in smiles_batch]).float().to(DEVICE)

        lossG = bce(fake_pred, torch.ones_like(fake_pred)) - torch.mean(rewards)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    #########################################
    # Save checkpoints
    #########################################

    if epoch % 20 == 0:
        torch.save(vae.state_dict(), f"vae_{epoch}.pt")
        torch.save(G.state_dict(), f"ganG_{epoch}.pt")
        print(f"Epoch {epoch} | VAE={loss_vae.item():.3f} | GAN={lossG.item():.3f}")

print("VAE–GAN–RL training complete.")
