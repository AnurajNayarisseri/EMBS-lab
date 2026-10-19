"""Sample molecules from a trained GRU policy checkpoint."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from rdkit import Chem
from .smiles_gru import SmilesVocabulary, SmilesGRUPolicy
from .rl import score_smiles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',default='data/training_smiles.csv')
    p.add_argument('--smiles-column',default='SMILES')
    p.add_argument('--out',default='results/generated_smiles.csv')
    p.add_argument('--n',type=int,default=1000)
    p.add_argument('--max-len',type=int,default=100)
    p.add_argument('--temperature',type=float,default=0.9)
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
    vocab=SmilesVocabulary([]); vocab.tokens=ck['tokens']; vocab.itos=vocab.tokens; vocab.stoi={t:i for i,t in enumerate(vocab.tokens)}
    vocab.pad_id=vocab.stoi['<pad>']; vocab.bos_id=vocab.stoi['<bos>']; vocab.eos_id=vocab.stoi['<eos>']; vocab.unk_id=vocab.stoi['<unk>']
    model=SmilesGRUPolicy(len(vocab.tokens),pad_id=vocab.pad_id).to(device); model.load_state_dict(ck['model']); model.eval()
    df=pd.read_csv(a.data); refs=df[a.smiles_column].dropna().astype(str).tolist(); rows=[]
    with torch.no_grad():
        remaining=a.n
        while remaining>0:
            n=min(a.batch_size,remaining); seq,_,_=model.sample(vocab.bos_id,vocab.eos_id,a.max_len,n,device=device,temperature=a.temperature)
            for ids in seq.cpu().tolist():
                smi=vocab.decode(ids); mol=Chem.MolFromSmiles(smi); rec=score_smiles(smi,refs)
                rows.append({'smiles':smi,'canonical_smiles':Chem.MolToSmiles(mol,canonical=True) if mol else '', 'valid':mol is not None,**rec})
            remaining-=n
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print(f'Wrote {len(rows)} samples to {out}; valid fraction={sum(r["valid"] for r in rows)/max(1,len(rows)):.3f}')
    print('Rewards are heuristic unless a validated external docking/scoring model is explicitly supplied.')
if __name__=='__main__': main()
