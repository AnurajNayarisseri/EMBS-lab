import selfies as sf
import numpy as np
from rdkit import Chem


# -----------------------------
# SMILES ↔ SELFIES Conversion
# -----------------------------
def smiles_to_selfies(smiles_list):
    selfies_list = []
    for smi in smiles_list:
        try:
            selfies_list.append(sf.encoder(smi))
        except:
            continue
    return selfies_list


def selfies_to_smiles(selfies_list):
    smiles_list = []
    for s in selfies_list:
        try:
            smiles_list.append(sf.decoder(s))
        except:
            continue
    return smiles_list


# -----------------------------
# Validation
# -----------------------------
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def filter_valid_smiles(smiles_list):
    return [s for s in smiles_list if is_valid_smiles(s)]


# -----------------------------
# Tokenization
# -----------------------------
def split_selfies(selfies):
    return list(sf.split_selfies(selfies))


def build_vocab(selfies_list):
    vocab = set()
    
    for s in selfies_list:
        tokens = split_selfies(s)
        vocab.update(tokens)
    
    vocab = sorted(list(vocab))
    
    # Special tokens
    vocab = ["<pad>", "<start>", "<end>"] + vocab
    
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = {i: s for s, i in stoi.items()}
    
    return stoi, itos


# -----------------------------
# Encoding / Decoding
# -----------------------------
def encode_selfies(selfies, stoi, max_len=100):
    tokens = ["<start>"] + split_selfies(selfies) + ["<end>"]
    
    token_ids = [stoi.get(t, 0) for t in tokens]
    
    if len(token_ids) < max_len:
        token_ids += [stoi["<pad>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    
    return token_ids


def decode_tokens(token_ids, itos):
    tokens = [itos.get(i, "") for i in token_ids]
    
    # remove special tokens
    tokens = [t for t in tokens if t not in ["<pad>", "<start>", "<end>"]]
    
    return "".join(tokens)


# -----------------------------
# Dataset Preparation
# -----------------------------
def prepare_dataset(smiles_list, max_len=100):
    # Convert to SELFIES
    selfies_list = smiles_to_selfies(smiles_list)
    
    # Build vocab
    stoi, itos = build_vocab(selfies_list)
    
    # Encode
    encoded = [encode_selfies(s, stoi, max_len) for s in selfies_list]
    
    return np.array(encoded), stoi, itos


# -----------------------------
# Batch Generator (for training)
# -----------------------------
def get_batches(data, batch_size=32):
    np.random.shuffle(data)
    
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# -----------------------------
# Debug / Example Run
# -----------------------------
if __name__ == "__main__":
    sample_smiles = ["CCO", "CCN", "CCC"]
    
    print("Original SMILES:", sample_smiles)
    
    selfies = smiles_to_selfies(sample_smiles)
    print("SELFIES:", selfies)
    
    reconstructed = selfies_to_smiles(selfies)
    print("Reconstructed SMILES:", reconstructed)
    
    data, stoi, itos = prepare_dataset(sample_smiles)
    
    print("Encoded shape:", data.shape)
    print("Vocab size:", len(stoi))