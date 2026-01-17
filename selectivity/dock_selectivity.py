import os
import subprocess
import pandas as pd

KINASES = {
 "CDK2":"1AQ1.pdb","CDK1":"6GU2.pdb","CDK4":"2W96.pdb","CDK7":"1UA2.pdb","CDK9":"4BCF.pdb",
 "EGFR":"1M17.pdb","BRAF":"1UWH.pdb","AKT1":"4EKL.pdb","ERK2":"2ERK.pdb","PI3Kγ":"3APC.pdb",
 "SRC":"2SRC.pdb","ABL1":"1OPJ.pdb","VEGFR2":"2OH4.pdb","JAK2":"4BBE.pdb","BTK":"5P9J.pdb"
}

LIGAND = "../docking/cdkcrc_embs.pdbqt"

def clean(pdb):
    out = pdb.replace(".pdb","_clean.pdb")
    os.system(f"grep -v HETATM {pdb} > {out}")
    return out

def dock(rec):
    cmd=["gnina","-r",rec,"-l",LIGAND,"--autobox_ligand",rec,"--score_only"]
    out=subprocess.check_output(cmd).decode()
    for l in out.split("\n"):
        if "CNNscore" in l:
            return float(l.split()[-1])
    return None

results=[]
for k,p in KINASES.items():
    print("Docking",k)
    rec=clean(p)
    s=dock(rec)
    results.append([k,s])

df=pd.DataFrame(results,columns=["Kinase","CNN_Affinity"])
cdk2=df[df.Kinase=="CDK2"]["CNN_Affinity"].values[0]
df["Relative_to_CDK2"]=df["CNN_Affinity"]/cdk2
df.sort_values("Relative_to_CDK2",ascending=False,inplace=True)
df.to_csv("kinome_profile.csv",index=False)

print(df)
