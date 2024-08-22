import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Self, Callable


dropout = 0.3
n_layer = 8
n_embd = 512
n_head = 8
block_size = 256

# Open dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars: list = sorted(list(set(text)))
vocab_size: int = len(chars) # number of total characters

stoi: dict = {ch:i for i, ch in enumerate(chars)} # dictionary to convert char to an index
itos: dict = {i:ch for i, ch in enumerate(chars)} # dictionary to convert index to a char
encode: Callable[[str], list] = lambda s: [stoi[c] for c in s] # encode an input str
decode: Callable[[list[int]], str] = lambda l: ''.join([itos[i] for i in l]) # decode input list

class Head(nn.Module):
    
    """ Single self attention head """

    def __init__(self: Self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # used for masked_fill

        self.dropout = nn.Dropout(dropout)
    
    def forward(self: Self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    
    """ Create multiple heads based on the Head class"""

    def __init__(self: Self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self: Self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    
    """ Basic MLP network """

    def __init__(self: Self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self: Self, x: Tensor) -> Tensor:
        return self.net(x)

class Block(nn.Module):
    
    """ Single block of self-attention + feedforward"""

    def __init__(self: Self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self: Self, x: Tensor) -> Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTModel(nn.Module):
    
    """ GPT model """

    def __init__(self: Self, device: str) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
           *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

        self.apply(self._kaiming_init)

    # use kaiming init for linear layers and normal for embedding layers
    def _kaiming_init(self: Self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self: Self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # generate characters based on the trained model 
    def generate(self: Self, idx: Tensor, max_new_tokens: int) -> None:
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            print(decode(idx_next.tolist()[0]), end='', flush=True)
            idx = torch.cat((idx, idx_next), dim=1)
        