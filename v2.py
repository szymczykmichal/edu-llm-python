import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

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


class Head(nn.Module):
    """ One head of self-attention

    > head_size in AIAYN is d_k or d_v and is equal to d_model/h, where h=8 is number of heads, 
        results in matrix size 512/8 = 64, here it is 32
    > n_embd is AIAYN is d_model = 512, here it is 32
    """

    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # (n_embd, head_size) (32, 32)
        self.query = nn.Linear(n_embd, head_size, bias=False) # (n_embd, head_size) (32, 32)
        self.value = nn.Linear(n_embd, head_size, bias=False) # (n_embd, head_size) (32, 32)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # block_size = context_length = 8

    def forward(self, x):
        """"
        wei = torch.einsum("bthd,bthd->bht", q, k) / (C ** 0.5) # (B, H, T, T)
        """
        B, T, C = x.shape # (B, T, C) (32, 8, 32)
        k = self.key(x) # (B, T, C) (32, 8, 32)
        q = self.query(x) # (B, T, C) (32, 8, 32)

        # calculate selt-attention pattern
        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (32, 8, 32) @ (32, 32, 8) -> (32, 8, 8)
        wei - wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x) # (B, T, C) (32, 8, 32)
        out = wei @ v # (32, 8, 8) @ (32, 8, 32) -> (32, 8, 32) [*]
        return out
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (vocab_size, n_embd) (65, 32)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (block_size, n_embd) (8, 32)
        self.sa_head = Head(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size) # (n_embd, vocab_size) (32, 65)

    def forward(self, idx, targets=None): # n_embd =-> C
        B, T = idx.shape # (B, T) (32, 8)
     
        tok_emb = self.token_embedding_table(idx) # (B, T, C), (B, block_size, n_embd), (32, 8, 32)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd) (block_size, n_embd), (8, 32)
        x = tok_emb + pos_emb # (B, T, n_embd) (B, block_size, n_embd), (32, 8, 32)
        x = self.sa_head(x) # (B, T, n_embd) (B, block_size, n_embd), (32, 8, 32) - one head of self-attention [*]
        logits = self.lm_head(tok_emb) # (B, T, vocab_size) (B, block_size, vocab_size), (32, 8, 65)


        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape # (B, T, vocab) (32, 8, 65)
            logits = logits.view(B*T, C) # [32, 65] -> token id distribution
            targets = targets.view(B*T) # [32] -> token id
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T, C) array of indices of the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #keep only the last block_size tokens
            logits, loss = self(idx_cond) #run prediction
            logits = logits[:, -1, :] #fetch last prediciton from the time dimension
            probs = F.softmax(logits, dim=-1) #(B, C), (B, probabilities)
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1), losowa probka
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel()
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