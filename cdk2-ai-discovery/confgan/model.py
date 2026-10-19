import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#############################################
#  Motif-based Molecular Graph Neural Network
#############################################

class MMGNN(nn.Module):
    """
    Motif-based Molecular GNN used as conditional encoder
    Nodes = pharmacophoric motifs
    Edges = chemical connectivity
    """
    def __init__(self, node_dim=32, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (batch, n_motifs, node_dim)
        h = F.relu(self.fc1(x))
        out, _ = self.gru(h)
        return out[:, -1, :]   # molecular embedding


#############################################
#  ConfGAN Generator
#############################################

class ConfGenerator(nn.Module):
    """
    Learns molecular distance–energy landscapes
    Outputs interatomic distance matrix + energy
    """
    def __init__(self, z_dim=128, cond_dim=128, n_atoms=32):
        super().__init__()
        self.n_atoms = n_atoms
        self.fc = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_atoms*n_atoms + 1)  # distances + energy
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        out = self.fc(x)
        D = out[:, :-1].view(-1, self.n_atoms, self.n_atoms)
        E = out[:, -1]
        D = (D + D.transpose(1,2)) / 2
        D = torch.abs(D)
        eye = torch.eye(self.n_atoms, device=D.device, dtype=D.dtype).unsqueeze(0)
        D = D * (1.0 - eye)
        return D, E


#############################################
#  ConfGAN Discriminator
#############################################

class ConfDiscriminator(nn.Module):
    """
    Judges if a distance-energy pair is real or generated
    """
    def __init__(self, n_atoms=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_atoms*n_atoms + 1, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, D, E):
        x = torch.cat([D.view(D.size(0), -1), E.unsqueeze(1)], dim=1)
        return self.fc(x)


#############################################
#  Distance Geometry Reconstruction
#############################################

def distance_to_xyz(D):
    """
    Euclidean Distance Geometry reconstruction
    Converts distance matrix to 3D coordinates
    """
    n = D.shape[0]
    J = torch.eye(n, device=D.device, dtype=D.dtype) - torch.ones(n,n,device=D.device,dtype=D.dtype)/n
    B = -0.5 * J @ (D**2) @ J
    eigvals, eigvecs = torch.linalg.eigh(B)
    coords = eigvecs[:, -3:] * torch.sqrt(torch.clamp(eigvals[-3:], min=1e-6))
    return coords


#############################################
#  Full ConfGAN Model
#############################################

class ConfGAN(nn.Module):
    """
    MM-GNN conditioned ConfGAN used in your paper
    """
    def __init__(self, n_atoms=32):
        super().__init__()
        self.mmg = MMGNN()
        self.G = ConfGenerator(n_atoms=n_atoms)
        self.D = ConfDiscriminator(n_atoms=n_atoms)

    def generate_conformer(self, motif_tensor):
        cond = self.mmg(motif_tensor)
        z = torch.randn(cond.size(0), 128, device=cond.device)
        D, E = self.G(z, cond)
        xyz = [distance_to_xyz(D[i]) for i in range(D.size(0))]
        return xyz, E


#############################################
#  Energy Loss (LJ + Harmonic)
#############################################

def energy_loss(D):
    # Stable, dimensionless regularizer only; not a force-field energy.
    n = D.shape[-1]
    mask = ~torch.eye(n, device=D.device, dtype=torch.bool)
    pair = D[..., mask].clamp(min=0.5, max=20.0)
    repulsion = torch.mean(torch.relu(1.5 - pair) ** 2)
    symmetry = torch.mean((D - D.transpose(-1, -2)) ** 2)
    diagonal = torch.diagonal(D, dim1=-2, dim2=-1).pow(2).mean()
    return repulsion + 0.1 * symmetry + 0.1 * diagonal
