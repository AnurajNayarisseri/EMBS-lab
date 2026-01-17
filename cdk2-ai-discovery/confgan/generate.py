import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from model import ConfGAN

#############################################
# Load trained ConfGAN
#############################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConfGAN().to(DEVICE)
model.load_state_dict(torch.load("confgan_epoch_1000.pt", map_location=DEVICE))
model.eval()

#############################################
# Load motif tensors
#############################################

motifs = np.load("../data/motifs.npy")
motifs = torch.tensor(motifs).float().to(DEVICE)

#############################################
# Generate conformers
#############################################

xyz_list, energy = model.generate_conformer(motifs)

#############################################
# Convert to RDKit molecules
#############################################

def xyz_to_mol(xyz, template_smi):
    mol = Chem.MolFromSmiles(template_smi)
    mol = Chem.AddHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, xyz[i].tolist())
    mol.AddConformer(conf)
    return mol

#############################################
# Save conformers
#############################################

template_smiles = "COc1nc(NC2CCN(CC2)C)c2ncnc(NC(=O)Nc3ccc(NC)cc3)c12"

writer = Chem.SDWriter("../results/confgan_conformers.sdf")

for i, coords in enumerate(xyz_list):
    mol = xyz_to_mol(coords.cpu().detach().numpy(), template_smiles)
    mol.SetProp("Energy", str(energy[i].item()))
    writer.write(mol)

writer.close()

print("ConfGAN conformers written to results/confgan_conformers.sdf")
