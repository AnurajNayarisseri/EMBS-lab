import unittest
import torch
from vae_gan_rl.vae import MolecularVAE,vae_loss
from vae_gan_rl.gan import MolGenerator,MolDiscriminator
from vae_gan_rl.smiles_gru import SmilesVocabulary,SmilesGRUPolicy
from vae_gan_rl.rl import score_smiles
from confgan.model import ConfGAN,energy_loss

class ModelSmokeTests(unittest.TestCase):
 def test_vae(self):
  x=torch.randint(0,2,(3,64)).float();m=MolecularVAE(64,8);y,mu,lv=m(x);self.assertEqual(y.shape,x.shape);self.assertTrue(torch.isfinite(vae_loss(x,y,mu,lv)))
 def test_gan(self):
  g=MolGenerator(8,64);d=MolDiscriminator(64);x=g(torch.randn(2,8));self.assertEqual(d(x).shape,(2,1))
 def test_gru_and_reinforce_gradient(self):
  v=SmilesVocabulary(['CCO','c1ccccc1']);m=SmilesGRUPolicy(len(v.tokens));seq,logps,ent=m.sample(v.bos_id,v.eos_id,max_len=8,n=2);loss=-logps.mean();loss.backward();self.assertIsNotNone(m.head.weight.grad)
 def test_reward(self):
  self.assertTrue(score_smiles('CCO')['valid']);self.assertFalse(score_smiles('not_smiles')['valid'])
 def test_confgan(self):
  m=ConfGAN(n_atoms=8); motifs=torch.randn(2,8,32);cond=m.mmg(motifs);d,e=m.G(torch.randn(2,128),cond);self.assertEqual(d.shape,(2,8,8));self.assertTrue(torch.isfinite(energy_loss(d)))
if __name__=='__main__':unittest.main()
