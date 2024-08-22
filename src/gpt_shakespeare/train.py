""" Script to train the shakespeare GPT model on the tiny shakespeare dataset"""

# Imports
import os
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from argparse import ArgumentParser

from gpt import encode, decode, block_size, GPTModel, text


# parse some args 
parser = ArgumentParser()
parser.add_argument('-b', '--batchsize', type=int, default=64, help='batch size to train on')
parser.add_argument('-i', '--iters', type=int, default=5000, help='num of iterations to run on')
parser.add_argument('-e', '--evalinterval', type=int, default=500, help='at which training iteration to evaluate')
parser.add_argument('-o', '--evaliters', type=int, default=500, help='how many iterations to eval on')
parser.add_argument('-l', '--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('-d', '--device', type=str, default='cuda', help='device on which train | use cuda if you have a cuda gpu')
parser.add_argument('-s', '--save', action='store_true', default=False, help='save the weights')
parser.add_argument('-p', '--path', type=str, default='./weights', help='where to save if save flag is used')
args = parser.parse_args()

# Hyperparameters
batch_size = args.batchsize
max_iters = args.iters
eval_interval = args.evalinterval
learning_rate = args.lr
eval_iters = 50
device = args.device

terminal_size = os.get_terminal_size()
terminal_width = terminal_size.columns

if not torch.cuda.is_available() and device == 'cuda':
    raise NotImplementedError('cuda is not availabe on your machine')

print(f'Running on device {device.upper()}'.center(terminal_width))
print(''.center(terminal_width, '='))

data: Tensor = torch.tensor(encode(text), dtype=torch.long)

# create train/val split 
n: int = int(0.9*len(data))
train_data: Tensor = data[:n]
val_data: Tensor = data[n:]

# create a batch based on batch_size
def get_batch(split: str) -> tuple[Tensor, Tensor]:
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    
# create model object
model: nn.Module = GPTModel(device=device)
m = model.to(device) # put device on cuda
 
# use torch no_grad to not store the grads when doing computations 
@torch.no_grad()
def estimate_loss() -> dict:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

if __name__ == '__main__':

    

    # training loop 
    with tqdm(total=max_iters // 10) as pgbar:

        for iter in range(max_iters):  

            pgbar.update(1)

            if iter % eval_interval == eval_interval-1:
                losses = estimate_loss()
                print(f'\n Step {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
                pgbar.reset()

            xb, yb = get_batch('train')

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))

    if args.save:
        assert args.path is not None, 'No path was given to save the model weights'
        # save the model weights
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(model.state_dict(), args.path + '/gpt_shakespeare' + str(np.random.randint(0, 200)) + '.pt')