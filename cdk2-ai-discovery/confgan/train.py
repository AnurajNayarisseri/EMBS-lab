"""Train the supplied ConfGAN-style model on distance-matrix data."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .model import ConfGAN,energy_loss

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--outdir',default='results/confgan');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--max-atoms',type=int,default=32);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 torch.manual_seed(a.seed);np.random.seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); data=np.load(a.data,allow_pickle=False)
 required={'motifs','distances','energies'}
 if not required.issubset(data.files): raise ValueError(f'Dataset must contain {required}')
 motifs=torch.tensor(data['motifs'],dtype=torch.float32); distances=torch.tensor(data['distances'],dtype=torch.float32); energies=torch.tensor(data['energies'],dtype=torch.float32)
 if motifs.ndim!=3 or motifs.shape[-1]!=32: raise ValueError('motifs must have shape (n, max_atoms, 32)')
 if distances.shape[1:]!=(a.max_atoms,a.max_atoms): raise ValueError('distance matrix shape does not match --max-atoms')
 loader=DataLoader(TensorDataset(motifs,distances,energies),batch_size=a.batch_size,shuffle=True); model=ConfGAN(n_atoms=a.max_atoms).to(device); bce=nn.BCELoss(); optg=torch.optim.Adam(list(model.G.parameters())+list(model.mmg.parameters()),lr=a.lr,betas=(.5,.999)); optd=torch.optim.Adam(model.D.parameters(),lr=a.lr,betas=(.5,.999)); out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);hist=[]
 for epoch in range(1,a.epochs+1):
  sums=[]
  for m,d,e in loader:
   m=m.to(device);d=d.to(device);e=e.to(device); b=len(m); cond=model.mmg(m); z=torch.randn(b,128,device=device); df,ef=model.G(z,cond)
   # Real/fake discriminator
   pr=model.D(d,e); pf=model.D(df.detach(),ef.detach()); ld=bce(pr,torch.ones_like(pr))+bce(pf,torch.zeros_like(pf));optd.zero_grad();ld.backward();optd.step()
   cond=model.mmg(m); z=torch.randn(b,128,device=device); df,ef=model.G(z,cond); pg=model.D(df,ef); adv=bce(pg,torch.ones_like(pg))
   # Pairwise distance loss anchors generated matrices to reference geometries.
   geom=torch.nn.functional.smooth_l1_loss(df,d); eloss=torch.nn.functional.smooth_l1_loss(ef,e)
   phys=energy_loss(df)
   lg=adv+geom+0.1*eloss+0.001*phys;optg.zero_grad();lg.backward();torch.nn.utils.clip_grad_norm_(list(model.G.parameters())+list(model.mmg.parameters()),5.0);optg.step();sums.append([ld.item(),lg.item(),geom.item(),eloss.item()])
  arr=np.mean(sums,axis=0).tolist(); row={'epoch':epoch,'loss_d':arr[0],'loss_g':arr[1],'distance_loss':arr[2],'energy_loss':arr[3]};hist.append(row)
  if epoch==1 or epoch%10==0 or epoch==a.epochs:
   print(json.dumps(row));torch.save({'model':model.state_dict(),'max_atoms':a.max_atoms,'epoch':epoch,'data':a.data},out/'confgan.pt')
 pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False);(out/'metadata.json').write_text(json.dumps({'dataset':a.data,'samples':len(motifs),'device':str(device),'warning':'Architecture implementation is not proof of reproducing original ConfGAN results. Verify feature semantics and original data/weights.'},indent=2))
if __name__=='__main__':main()
