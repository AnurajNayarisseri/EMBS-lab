import random
import numpy as np
from collections import defaultdict
from src.reward import reward


# -----------------------------
# RL Agent (Q-learning style)
# -----------------------------
class MoleculeRLAgent:
    def __init__(self, epsilon=0.2, alpha=0.1):
        self.q_table = defaultdict(float)
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha      # learning rate

    def choose_action(self, state):
        # epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(state)
        else:
            return max(state, key=lambda x: self.q_table[x])

    def update(self, action, reward_value):
        self.q_table[action] += self.alpha * (reward_value - self.q_table[action])


# -----------------------------
# Mutation Function (Action Space)
# -----------------------------
def mutate_smiles(smiles):
    """Simple mutation operator (placeholder for real chemistry ops)"""
    if len(smiles) < 2:
        return smiles
    
    smi_list = list(smiles)
    i, j = random.sample(range(len(smi_list)), 2)
    smi_list[i], smi_list[j] = smi_list[j], smi_list[i]
    
    return "".join(smi_list)


# -----------------------------
# RL Optimization Loop
# -----------------------------
def optimize(smiles_list, iterations=100, epsilon=0.2):
    agent = MoleculeRLAgent(epsilon=epsilon)
    
    best_molecules = []

    for _ in range(iterations):
        # Select initial molecule
        state = smiles_list
        current = agent.choose_action(state)

        # Apply mutation (action)
        new_smiles = mutate_smiles(current)

        # Get reward
        r = reward(new_smiles)

        # Update agent
        agent.update(new_smiles, r)

        # Store good molecules
        if r > 1:
            best_molecules.append((new_smiles, r))

    # Sort best molecules
    best_molecules = list(set(best_molecules))
    best_molecules = sorted(best_molecules, key=lambda x: -x[1])

    return best_molecules


# -----------------------------
# Policy Gradient (Optional Advanced)
# -----------------------------
class PolicyNetwork:
    def __init__(self):
        self.memory = []

    def store(self, smiles, reward_value):
        self.memory.append((smiles, reward_value))

    def get_top(self, top_k=10):
        return sorted(self.memory, key=lambda x: -x[1])[:top_k]


def optimize_policy(smiles_list, iterations=100):
    policy = PolicyNetwork()

    for _ in range(iterations):
        smi = random.choice(smiles_list)
        mutated = mutate_smiles(smi)

        r = reward(mutated)
        policy.store(mutated, r)

    return policy.get_top()


# -----------------------------
# Batch Optimization
# -----------------------------
def batch_optimize(smiles_list, batch_size=50, rounds=5):
    all_best = []

    for _ in range(rounds):
        subset = random.sample(smiles_list, min(batch_size, len(smiles_list)))
        best = optimize(subset, iterations=50)
        all_best.extend(best)

    # Remove duplicates and sort
    all_best = list(set(all_best))
    all_best = sorted(all_best, key=lambda x: -x[1])

    return all_best


# -----------------------------
# Debug Run
# -----------------------------
if __name__ == "__main__":
    sample_smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]

    print("Running RL optimization...")
    best = optimize(sample_smiles, iterations=50)

    print("\nTop molecules:")
    for smi, score in best[:10]:
        print(smi, score)