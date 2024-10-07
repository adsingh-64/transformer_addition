import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 6  # max context length is 3 digits plus 3 digits = 6 digits
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 20  # 0 through 19
# ------------

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(1337)

"""serve a + b = c for a and b in {0, ... , 999}"""
def get_problem():
    a = torch.randint(0, 1000, (1,))
    b = torch.randint(0, 1000, (1,))
    c = a + b
    a_digits_reversed = (["0"] * (3 - len(str(a.item()))) + list(str(a.item())))[::-1]  # pad tens or hundreds place with zeros if necessary
    b_digits_reversed = (["0"] * (3 - len(str(b.item()))) + list(str(b.item())))[::-1]
    c_digits = list(str(c.item()))
    if len(c_digits) == 4:
        c_digits[0] = c_digits[0] + c_digits[1]
        del c_digits[1]
    c_digits_reversed = (["0"] * (3 - len(c_digits)) + c_digits)[::-1]
    x = [digit for place in zip(a_digits_reversed, b_digits_reversed) for digit in place]  # 6 digits
    y = ["20"] * 6  # 20 not in vocab, used for ignore index
    for i in range(1, 6, 2):
        y[i] = c_digits_reversed[i // 2]
    x_out = torch.tensor([int(ch) for ch in x])
    y_out = torch.tensor([int(ch) for ch in y])
    x_out, y_out = x_out.to(device), y_out.to(device)
    return x_out, y_out


class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        wei = (q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5)  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T) 
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, head_size*num_heads)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """linear layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # [B, T, n_embd]
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # transition matrix to prepare for going back into residual pathway via addition
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # departure from Attention is All You Need -- we apply LN before transformation
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (block_size, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (block_size, n_embd)
        x = tok_emb + pos_emb  #  (block_size, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (block_size, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, _ = logits.shape
            logits = logits.view(B * T, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=20)
        return logits, loss


model = LanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # sample a batch of data
    x_list, y_list = [], []

    # Sample multiple problems to form a batch
    for _ in range(batch_size):
        x, y = get_problem()
        x_list.append(x)
        y_list.append(y)

    # Stack the lists into tensors
    xb = torch.stack(x_list)
    yb = torch.stack(y_list)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"step {iter}: {loss.item()}")

# inference
count = 0
for i in range(10000):
    # serve problem
    a = torch.randint(0, 999, (1,))
    b = torch.randint(0, 999, (1,))
    c = a + b
    a_digits_reversed = (["0"] * (3 - len(str(a.item()))) + list(str(a.item())))[::-1]  # pad tens or hundreds place with zeros if necessary
    b_digits_reversed = (["0"] * (3 - len(str(b.item()))) + list(str(b.item())))[::-1]
    c_digits = list(str(c.item()))
    if len(c_digits) == 4:
        c_digits[0] = c_digits[0] + c_digits[1]
        del c_digits[1]
    c_digits_reversed = (["0"] * (3 - len(c_digits)) + c_digits)[::-1]

    x = [digit for place in zip(a_digits_reversed, b_digits_reversed) for digit in place]  # 6 digits
    y = ["0"] * 6

    for i in range(1, 6, 2):
        y[i] = c_digits_reversed[i // 2]

    x_out = torch.tensor([int(ch) for ch in x]).view([1, 6])
    y_out = torch.tensor([int(ch) for ch in y])
    x_out, y_out = x_out.to(device), y_out.to(device)

    # forward pass
    logits, _ = model(x_out, y_out)
    mask = torch.arange(6) % 2 == 1
    masked = logits[mask, :]
    output = ""
    for i in range(3):
        output += str(torch.argmax(masked[-(i + 1), :]).item())
    answer = int(output)
    if c.item() == answer:
        count += 1

accuracy = count / 10000
print(f"Accuracy is {accuracy:.2f}")

"""
Training on 1x A100 GPU for 10 minutes achieves 99.9% accuracy
"""