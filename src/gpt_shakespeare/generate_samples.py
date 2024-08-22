from argparse import ArgumentParser
import torch 

import gpt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = ArgumentParser()
parser.add_argument('-t', '--tokens', type=int, default=10000, help='pass the number of tokens to generate')
parser.add_argument('-p', '--path', type=str, default='./weights', help='where to load weights from')
args = parser.parse_args()

model = gpt.GPTModel(device=device)
m = model.to(device=device)
model.load_state_dict(torch.load('./weights/gpt_shakespeare.pt', weights_only=True))
model.eval()

if __name__ == '__main__':
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.generate(context, max_new_tokens=args.tokens)