"""GNINA screening wrapper. Requires GNINA, Open Babel, and prepared structures.
Use a ligand-bound reference ligand to define the autobox; never use the full
receptor as --autobox_ligand.
"""
import argparse,shutil,subprocess,tempfile
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def require(cmd):
 if shutil.which(cmd) is None: raise RuntimeError(f'{cmd} not found on PATH; install it separately.')

def smiles_to_pdbqt(smiles,out_path):
 mol=Chem.MolFromSmiles(str(smiles))
 if mol is None: raise ValueError(f'Invalid SMILES: {smiles}')
 mol=Chem.AddHs(mol)
 if AllChem.EmbedMolecule(mol,randomSeed=42)!=0: raise RuntimeError('RDKit conformer generation failed')
 if AllChem.MMFFOptimizeMolecule(mol)!=0: AllChem.UFFOptimizeMolecule(mol)
 molfile=out_path.with_suffix('.sdf'); w=Chem.SDWriter(str(molfile));w.write(mol);w.close()
 subprocess.run(['obabel',str(molfile),'-O',str(out_path)],check=True,capture_output=True,text=True)
 return out_path

def parse_gnina(text):
 # GNINA output tables usually include CNNscore and CNNaffinity columns.
 lines=text.splitlines();
 for i,line in enumerate(lines):
  if 'CNNscore' in line and 'CNNaffinity' in line:
   headers=line.split()
   for row in lines[i+1:]:
    vals=row.split()
    if len(vals)>=len(headers):
     try:
      d=dict(zip(headers,vals)); return float(d.get('CNNscore','nan')),float(d.get('CNNaffinity','nan'))
     except ValueError: continue
 return None,None

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='CSV with Name and SMILES columns');p.add_argument('--receptor',required=True,help='Prepared receptor PDBQT');p.add_argument('--reference-ligand',required=True,help='PDBQT/SDF/MOL reference ligand defining binding-site box');p.add_argument('--outdir',default='results/gnina');p.add_argument('--exhaustiveness',type=int,default=16);p.add_argument('--num-modes',type=int,default=10);a=p.parse_args()
 require('gnina');require('obabel'); inp=pd.read_csv(a.input)
 for c in ['Name','SMILES']:
  if c not in inp: raise ValueError(f'Input CSV needs column {c}')
 receptor=Path(a.receptor).resolve(); ref=Path(a.reference_ligand).resolve()
 if not receptor.is_file() or not ref.is_file(): raise FileNotFoundError('Receptor or reference ligand file does not exist')
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True); rows=[]
 for i,r in inp.iterrows():
  name=str(r['Name']); safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in name)[:80] or f'ligand_{i}'
  try:
   ligand=smiles_to_pdbqt(str(r['SMILES']),out/f'{safe}.pdbqt')
   cmd=['gnina','-r',str(receptor),'-l',str(ligand),'--autobox_ligand',str(ref),'--num_modes',str(a.num_modes),'--exhaustiveness',str(a.exhaustiveness)]
   proc=subprocess.run(cmd,capture_output=True,text=True,check=True); score,aff=parse_gnina(proc.stdout+'\n'+proc.stderr)
   rows.append({'Name':name,'SMILES':r['SMILES'],'CNNscore':score,'CNNaffinity':aff,'status':'ok'})
  except Exception as e: rows.append({'Name':name,'SMILES':r['SMILES'],'CNNscore':None,'CNNaffinity':None,'status':f'error: {e}'})
  print(f'[{i+1}/{len(inp)}] {name}: {rows[-1]["status"]}')
 pd.DataFrame(rows).to_csv(out/'gnina_results.csv',index=False)
if __name__=='__main__':main()
