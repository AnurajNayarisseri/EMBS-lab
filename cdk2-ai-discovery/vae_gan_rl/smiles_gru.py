"""Character-level autoregressive GRU policy for SMILES sequence generation."""
import torch
from torch import nn

SPECIAL=['<pad>','<bos>','<eos>','<unk>']
class SmilesVocabulary:
    def __init__(self, strings):
        chars=sorted(set(''.join(str(s) for s in strings)))
        self.tokens=SPECIAL+chars; self.stoi={t:i for i,t in enumerate(self.tokens)}; self.itos=self.tokens
        self.pad_id=self.stoi['<pad>']; self.bos_id=self.stoi['<bos>']; self.eos_id=self.stoi['<eos>']; self.unk_id=self.stoi['<unk>']
    def encode(self,s,max_len=128):
        ids=[self.bos_id]+[self.stoi.get(c,self.unk_id) for c in str(s)[:max_len-2]]+[self.eos_id]
        return ids
    def decode(self,ids):
        out=[]
        for i in ids:
            t=self.itos[int(i)]
            if t=='<eos>': break
            if t not in SPECIAL: out.append(t)
        return ''.join(out)

class SmilesGRUPolicy(nn.Module):
    def __init__(self,vocab_size,emb_dim=128,hidden_dim=256,pad_id=0):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=pad_id); self.gru=nn.GRU(emb_dim,hidden_dim,batch_first=True); self.head=nn.Linear(hidden_dim,vocab_size)
    def forward(self,ids,hidden=None):
        x=self.embedding(ids); y,hidden=self.gru(x,hidden); return self.head(y),hidden
    def sample(self,bos_id,eos_id,max_len=100,n=32,device='cpu',temperature=1.0):
        self.eval(); token=torch.full((n,1),bos_id,dtype=torch.long,device=device); hidden=None; seq=[]; logps=[]; entropies=[]; alive=torch.ones(n,dtype=torch.bool,device=device)
        for _ in range(max_len):
            logits,hidden=self.forward(token,hidden); logits=logits[:,-1,:]/max(temperature,1e-4); dist=torch.distributions.Categorical(logits=logits); nxt=dist.sample(); logps.append(dist.log_prob(nxt)); entropies.append(dist.entropy()); seq.append(nxt); alive=alive & (nxt!=eos_id); token=nxt[:,None]
            if not alive.any(): break
        seq=torch.stack(seq,dim=1) if seq else torch.empty((n,0),dtype=torch.long,device=device)
        return seq,torch.stack(logps,dim=1),torch.stack(entropies,dim=1)
