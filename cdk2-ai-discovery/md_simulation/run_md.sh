#!/bin/bash

##############################################
# GROMACS 2024 MD PIPELINE
# CDK2 – CDKCRC-EMBS Complex
##############################################

set -e

echo "=== CDK2 MD simulation starting ==="

##############################################
# 1. Energy Minimization
##############################################

cat > em.mdp << EOF
integrator = steep
nsteps = 50000
emtol = 1000.0
emstep = 0.01
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
constraints = h-bonds
EOF

gmx grompp -f em.mdp -c neutral.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em

##############################################
# 2. NVT Equilibration (300 K)
##############################################

cat > nvt.mdp << EOF
integrator = md
nsteps = 50000
dt = 0.002
tcoupl = nose-hoover
tc-grps = Protein_LIG Water_and_ions
tau_t = 1.0 1.0
ref_t = 300 300
pcoupl = no
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
EOF

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt

##############################################
# 3. NPT Equilibration (1 atm)
##############################################

cat > npt.mdp << EOF
integrator = md
nsteps = 50000
dt = 0.002
tcoupl = nose-hoover
tc-grps = Protein_LIG Water_and_ions
tau_t = 1.0 1.0
ref_t = 300 300
pcoupl = mt
pcoupltype = isotropic
tau_p = 5.0
ref_p = 1.0
compressibility = 4.5e-5
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
EOF

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr
gmx mdrun -deffnm npt

##############################################
# 4. 100 ns Production MD
##############################################

cat > md.mdp << EOF
integrator = md
nsteps = 50000000
dt = 0.002
tcoupl = nose-hoover
tc-grps = Protein_LIG Water_and_ions
tau_t = 1.0 1.0
ref_t = 300 300
pcoupl = mt
pcoupltype = isotropic
tau_p = 5.0
ref_p = 1.0
compressibility = 4.5e-5
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
nstxout-compressed = 5000
nstenergy = 1000
nstlog = 1000
EOF

gmx grompp -f md.mdp -c npt.gro -p topol.top -o md.tpr
gmx mdrun -deffnm md

echo "=== MD simulation complete ==="
