import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch:idx for idx, ch in enumerate(chars)}
itos = {idx:ch for ch,idx in stoi.items()}

def encode(text):
    return [stoi[ch] for ch in text]

def decode(nums):
    return "".join([itos[idx] for idx in nums])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*data.shape[0])
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #(B, T, C)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # [32, 65] -> token id distribution
            targets = targets.view(B*T) # [32] -> token id
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T, C) array of indices of the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx) #run prediction
            logits = logits[:, -1, :] #fetch last prediciton from the time dimension
            probs = F.softmax(logits, dim=-1) #(B, C), (B, probabilities)
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1), losowa probka
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter: {iter}, train: {losses['train']}, val: {losses['val']}")
        
    x, y = get_batch("train")

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(m.generate(torch.zeros([1,1], dtype=torch.long), 400)[0].tolist()))