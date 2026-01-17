import os
import subprocess

#############################################
# Input files
#############################################

RECEPTOR = "cdk2_clean.pdb"         # CDK2 (1AQ1 without ligand)
LIGAND = "cdkcrc_embs.pdb"          # CDKCRC-EMBS from LigParGen
COMPLEX = "complex.pdb"

#############################################
# Merge protein + ligand
#############################################

def merge_complex():
    with open(COMPLEX, "w") as out:
        out.write(open(RECEPTOR).read())
        out.write(open(LIGAND).read())

#############################################
# Run GROMACS system preparation
#############################################

def run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

merge_complex()

#############################################
# Protein topology
#############################################

run("gmx pdb2gmx -f complex.pdb -o complex.gro -ff oplsaa -water spce")

#############################################
# Define box
#############################################

run("gmx editconf -f complex.gro -o boxed.gro -c -d 1.0 -bt cubic")

#############################################
# Solvate
#############################################

run("gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top")

#############################################
# Add ions
#############################################

run("gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr")
run("echo SOL | gmx genion -s ions.tpr -o neutral.gro -p topol.top -neutral")

print("System prepared: neutral.gro, topol.top")
