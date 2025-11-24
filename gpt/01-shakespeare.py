# First check "makemore"

import sys, os, torch, torch.nn as nn
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import get_file

torch.manual_seed(1337)

text = get_file("shakespeare.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

print("Lenght of the dataset:", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("hi there"))
# print(decode(encode("hi there")))

data = torch.tensor(encode(text), dtype = torch.long)
# print(data.shape)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# hyperparameters
block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300 # nubmer of steps to do before printing progress
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200 # number batches to do model evaluation for every "print progress" stop
n_embd = 32 # embedding size (number of trainable characteristics per char)
# ----
print("Device:", device)

def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i   : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def print_examples(xb, yb):
    print("inputs:")
    print(xb.shape)
    print(xb)
    print("targets:")
    print(yb.shape)
    print(yb)
    print("---")
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b,t]
            print(context, "->", target)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B,T) tensor of integers
        # emb table plucks out a emb vector for every cell in idx matrix number (think about it as a 3rd dimension)

        # here we don't use any context in the model, i.e. the prediction (logits) is based on the character at a position only
        logits = self.token_embedding_table(idx) # (B, T, C), B - batch, T - time, C - channel

        if targets==None:
            loss = None
        else:
            # PyTorch wants (B, C, T), so we need to reshape
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
def generate(m, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        logits, _ = m(idx)
        # focus only on last time step
        logits = logits[:, -1, :] # becomes (B, C)
        probs = F.softmax(logits, dim=-1) # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

# If we create the model and do one forward step:

# model = BigramLanguageModel(vocab_size)
# m = model.to(device)

# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

# then we expect loss of about -ln(1/65) which is ~ 4.17
# current loss is 5.03

def generate_sample(m, max_new_tokens):
    idx = torch.zeros((1,1), dtype=torch.long, device=device) # (B, T) = (1, 1), 0 is newline
    print(decode(generate(m, idx, max_new_tokens)[0].tolist()))

# generate_sample()
# Model produces garbage because it is now (a) bigram, i.e. context not used (b) not trained
# Let's train it!
# In Makemore we used stochastic gradient descent (torch.optim.SGD), but AdamW is better.
# Default LR could be 1e-4, but our model is small, so we use larger value.

@torch.no_grad()
def estimate_loss(m):
    # This functions get several batches (eval_iters), calls the model, save the loss, and then get the mean from saved losses
    out = {}
    # set model to "eval" mode
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # revert model back to "train" mode
    m.train()
    return out

def train(m):
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")

        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# So we start on a simple Bigram model, we still get a bad result after training
# model = BigramLanguageModel(vocab_size)
# m = model.to(device)
# train(m)
# generate_sample(m, 500)

# Let's make the model more complex!

####
# Mathematical trick.
# But before that, let's see how can we create a "context", so a char can "interact" with previous chars.
# The simplest form of interaction is to have average between current and all previous.
# Note, that char is represented by a 3rd dimension vector, called Channel.

B,T,C = 4,8,2
x = torch.randn(B,T,C)

# We want x[b,t] = mean_{i<=t} x[b,i]

xbow = torch.zeros((B,T,C)) # bow means "bag of words"
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C)
        xbow[b,t] = torch.mean(xprev, 0)

# The same two loops can be done by multiplying original matrix by lower triangle matrix, where non-zero cells are one divided by correspondent row number.
# Example:
# Random matrix b:
# 2 7
# 6 4
# 6 5
# Lower triangle matrix a = torch.tril(torch.ones(3,3)) divided by torch.sum(a, 1, keepdim=True):
# 1 0 0 / 1   1   0   0
# 1 1 0 / 2 = 0.5 0.5 0
# 1 1 1 / 3   0.3 0.3 0.3 
# Then, c = a @ b:
# 2   7
# 4   5.5
# 4.7 5.3
# This is exactly what we did with 2 for loops.

# wei = torch.tril(torch.ones(T, T))
# wei = wei / wei.sum(1, keepdim=True)
# xbow2 = wei @ x  # (T,T) @ (B, T, C) ---> (B, T, T) @ (B, T, C) ---> (B, T, C)
# print(torch.allclose(xbow, xbow2)) # True

# And finally, another version using Softmax:

tril = torch.tril(torch.ones(T, T))
# 1 0 0
# 1 1 0
# 1 1 1
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf")) # at all places where TRIL cells are 0 - put -inf
# 1 -inf -inf
# 1  1   -inf
# 1  1    1
wei = F.softmax(wei, dim=-1)
# 1   0   0
# 0.5 0.5 0
# 0.3 0.3 0.3
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3)) # True

# Important: wei represents how much previous char affects current. Aka Affinity. And wei means weights.
# That's why we artificially set weights of future context chars to -inf, because they do not affect a current char.
# So we set it to zero now, but infact those are trainable weights.
# And so during training the model will set those weights to some learnable affinities between the chars.

####

# Improving the model
# Changing embedding size and adding a linear layer after embedding layer.

class BigramLanguageModel2(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B,T) tensor of integers
        # emb table plucks out a emb vector for every cell in idx matrix number (think about it as a 3rd dimension)

        # here we don't use any context in the model, i.e. the prediction (logits) is based on the character at a position only
        tok_emb = self.token_embedding_table(idx) # (B, T, C), B - batch, T - time, C - channel

        logits = self.lm_head(tok_emb) # (B, T, vocab_size)

        if targets==None:
            loss = None
        else:
            # PyTorch wants (B, C, T), so we need to reshape
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
model = BigramLanguageModel2(vocab_size)
m = model.to(device)
train(m)
generate_sample(m, 500)