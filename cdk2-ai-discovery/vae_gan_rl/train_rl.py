"""Train a separate character-GRU SMILES policy with REINFORCE."""
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from rdkit import Chem
from .smiles_gru import SmilesVocabulary,SmilesGRUPolicy
from .rl import score_smiles

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/training_smiles.csv'); p.add_argument('--smiles-column',default='SMILES'); p.add_argument('--outdir',default='results/rl'); p.add_argument('--steps',type=int,default=1000); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--max-len',type=int,default=100); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--entropy-weight',type=float,default=.01); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df=pd.read_csv(args.data); train=[str(s) for s in df[args.smiles_column].dropna() if Chem.MolFromSmiles(str(s)) is not None]
    if not train: raise ValueError('No valid training SMILES.')
    vocab=SmilesVocabulary(train); model=SmilesGRUPolicy(len(vocab.tokens),pad_id=vocab.pad_id).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); logs=[]; records=[]
    for step in range(1,args.steps+1):
        model.train(); seq,logps,ents=model.sample(vocab.bos_id,vocab.eos_id,args.max_len,args.batch_size,device=device)
        strings=[vocab.decode(row) for row in seq.cpu().tolist()]; rewards=np.asarray([score_smiles(s,train)['reward'] for s in strings],dtype=np.float32)
        # REINFORCE with batch-normalized advantages. Include discriminator reward only if a trained discriminator is integrated; none is fabricated here.
        adv=torch.tensor(rewards,device=device); adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
        mask=torch.ones_like(logps); policy_loss=-(logps*adv[:,None]*mask).sum(dim=1).mean(); entropy=ents.mean(); loss=policy_loss-args.entropy_weight*entropy
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        logs.append({'step':step,'loss':float(loss.detach()),'policy_loss':float(policy_loss.detach()),'entropy':float(entropy.detach()),'mean_reward':float(rewards.mean()),'valid_fraction':float(np.mean([Chem.MolFromSmiles(s) is not None for s in strings]))})
        if step==1 or step%10==0 or step==args.steps:
            print(json.dumps(logs[-1]));
            for s,r in zip(strings,rewards): records.append({'step':step,'smiles':s,'reward':float(r),'valid':Chem.MolFromSmiles(s) is not None})
            torch.save({'model':model.state_dict(),'tokens':vocab.tokens,'step':step,'config':vars(args)},out/'gru_policy.pt')
    pd.DataFrame(logs).to_csv(out/'training_history.csv',index=False); pd.DataFrame(records).to_csv(out/'sampled_smiles.csv',index=False); (out/'metadata.json').write_text(json.dumps({'training_smiles_count':len(train),'vocab_size':len(vocab.tokens),'device':str(device),'reward':'QED + transparent SA heuristic + max similarity proxy; similarity is not docking','note':'This is a standalone REINFORCE policy. No trained discriminator reward or Dockproxy model was supplied.'},indent=2))
if __name__=='__main__': main()
