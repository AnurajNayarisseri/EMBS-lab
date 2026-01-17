import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

#############################################
# Load MD Trajectory
#############################################

traj = md.load("md.xtc", top="md.gro")
protein = traj.topology.select("protein")
traj = traj.atom_slice(protein)

#############################################
# RMSD
#############################################

ref = traj[0]
rmsd = md.rmsd(traj, ref)

plt.plot(rmsd)
plt.xlabel("Frame")
plt.ylabel("RMSD (nm)")
plt.savefig("rmsd.png")
plt.clf()

#############################################
# RMSF
#############################################

rmsf = md.rmsf(traj, ref)

plt.plot(rmsf)
plt.xlabel("Residue")
plt.ylabel("RMSF (nm)")
plt.savefig("rmsf.png")
plt.clf()

#############################################
# Radius of Gyration
#############################################

rg = md.compute_rg(traj)
plt.plot(rg)
plt.xlabel("Frame")
plt.ylabel("Rg (nm)")
plt.savefig("rgyr.png")
plt.clf()

#############################################
# PCA
#############################################

X = traj.xyz.reshape(traj.n_frames, -1)
pca = PCA(n_components=2)
PC = pca.fit_transform(X)

plt.scatter(PC[:,0], PC[:,1], s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("pca.png")
plt.clf()

#############################################
# DCCM
#############################################

coords = traj.xyz
mean = coords.mean(axis=0)
delta = coords - mean
n = delta.shape[1]

dccm = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        num = np.mean(delta[:,i,:] * delta[:,j,:])
        den = np.sqrt(np.mean(delta[:,i,:]**2) * np.mean(delta[:,j,:]**2))
        dccm[i,j] = num/den

plt.imshow(dccm, cmap="bwr", vmin=-1, vmax=1)
plt.colorbar()
plt.savefig("dccm.png")
plt.clf()

#############################################
# Free Energy Landscape (RMSD vs Rg)
#############################################

x = rmsd
y = rg

xy = np.vstack([x,y])
kde = gaussian_kde(xy)
xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

F = -np.log(zi.reshape(xi.shape))

plt.contourf(xi, yi, F, 50, cmap="viridis")
plt.xlabel("RMSD (nm)")
plt.ylabel("Rg (nm)")
plt.colorbar(label="Free Energy")
plt.savefig("FEL.png")
plt.clf()

print("Analysis completed. RMSD, RMSF, PCA, DCCM and FEL generated.")
