import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from datetime import datetime
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Movie Plot GPT")
parser.add_argument('--checkpoint', type=str, default=None, help='Path to your checkpoint file')
parser.add_argument('--training_data', type=str, default='dataset/movie plots/all_plots.txt', help='Path to the training data file')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--context_length', type=int, default=256, help='Context length for predictions')
parser.add_argument('--max_iters', type=int, default=10000, help='Maximum number of iterations for training')
parser.add_argument('--eval_interval', type=int, default=500, help='How often to evaluate the model')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for the optimizer')
parser.add_argument('--n_embd', type=int, default=512, help='Number of dimensions for the embeddings')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of tokens to generate')
args = parser.parse_args()


# hyperparameters
batch_size = args.batch_size # how many independent sequences will we process in parallel? DROP IT DOWN IF CPU
block_size = args.context_length # context length DROP IT DOWN IF CPU
max_iters = args.max_iters # how many iterations to use for training
eval_interval = args.eval_interval # how often to evaluate the model
learning_rate = args.lr # learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many iters to use for evaluation of the loss
n_embd = args.n_embd # how many dimensions to use for the embeddings DROP IT DOWN IF CPU
n_head = 8 # how many heads for an attention block DROP IT DOWN IF CPU
n_layer = 8 # how many attention blocks to use in the model DROP IT DOWN IF CPU
dropout = 0.2 # dropout rate
max_new_tokens = args.max_new_tokens # how many new tokens to generate
# ------------

# seed for reproducibility
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(args.training_data, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # getting random starting indexes for our blocks
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y is just x but shifted 1 to right
    # we want to know the next element of each char in x
    # so we take the shifted version of x to store next element for each char
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # we need to move our data to the device that we will use for computation
    x, y = x.to(device), y.to(device)
    return x, y

# skipping gradient calculation for evaluation
# even though we don't call the backward function, PyTorch will still
# track the operations happening in the forward pass becuase those tensors
# have requires_grad=True. To prevent this tracking we can use this decorator
@torch.no_grad()
def estimate_loss():
    out = {}
    # we need to set the model to evaluation mode
    # dropout layers behave differently during evaluation
    # batch norm layers also behave differently during evaluation
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # set the model back to training mode
    model.train()
    return out

# one head of the self attention mechanism
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        #self.head_size = head_size
        
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # buffers are not modified by gradient updates
        # but they will be moved to the device that the model is on
        # and also they will be a part of the state dict of the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # B,T,head_size
        k = self.key(x) # B,T,head_size
        
        
        attn = q @ k.transpose(-2, -1) * C ** (-0.5) # B,T,T
        attn = attn.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        v = self.value(x) # B,T,head_size
        out = attn @ v # B,T,head_size
        return out 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        head_out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads * head_size)
        head_out = self.dropout(self.proj(head_out))
        return head_out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
# communication followed by computation
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # n_heads * head_size = n_embd
        self.sa = MultiHeadAttention(n_heads, n_embd//n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        # these cumulative operations are called residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token's value will represent the meaning of that token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each positional embedding tells the model where the token is in the sequence
        # without these model couldn't know the position of the token in the sequence
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        # we will use the self attention mechanism to learn the relationships between tokens
        # here's our attention blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_head) for _ in range(n_layer)])
        # this is where we get the logits for the next token out of meaning of the current token
        # For more info about logits check the simple_bigram.py
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # fetching the learned token embeddings for each token in the sequence
        token_embs = self.token_embedding_table(idx) # (B,T,n_embd)
        # fetching the learned positional embeddings for each position in the sequence
        pos_embs = self.positional_embedding(torch.arange(T, device=idx.device)) # (T,n_embd)
        # adding the token and positional embeddings together
        x = token_embs + pos_embs # (B,T,n_embd)
        # applying the self attention mechanism to the embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        # getting the logits for the next token out of the embeddings which represent the meaning
        logits = self.lm_head(x) # (B,T,C=vocab_size)

        if targets is None:
            # during generation we will not have targets
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # the line below will first apply softmax to our logits,
            # turning our logits into a probability distribution with a sum of 1
            # then we will take the the correct next token with the value
            # we have from the targets
            # then we will take the -log of the likelihood of the true next char
            # this will be our loss value
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the block size
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to last dimension to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # tokens are chosen based on the probability distribution
            # given by their likelihoods. this is called sampling. For example:
            # A=0.7,B=0.2,C=0.1
            # A would have 70% chance of being chosen
            # B would have 20%
            # C would have 10%
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if args.checkpoint is not None:
    print("Custom checkpoint file found: " + args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device), weights_only=True))
    print("Model loaded from the checkpoint")
    print(str(max_new_tokens) + " chars will be generated. You can change this with command line arguments.")
    start_str = encode(input("Enter the starting string: ") + ' ')
    start_idx = torch.tensor(start_str, dtype=torch.long).unsqueeze(0).to(device)
    output = decode(model.generate(start_idx, max_new_tokens=max_new_tokens)[0].tolist())
    print(output)
    with open(f"outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(output)

else:
    print("No checkpoints found. Training from scratch")
    print(args.training_data + " will be used for training")
    # training loop
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        # pytorch accumulates the gradients by default
        # to prevent that we need to set the gradients to zero
        # additionally we're setting them to none for optimization
        optimizer.zero_grad(set_to_none=True)

        # calculating the gradients from the loss tensor
        loss.backward()

        # after calculating the gradients we take one step in the direction of them
        optimizer.step()

    checkpointname = f"checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), checkpointname)
    print(f"Checkpoint saved: {checkpointname}")

    print("Training finished")

    start_str = encode(input("Enter the starting string: ") + ' ')
    start_idx = torch.tensor(start_str, dtype=torch.long).unsqueeze(0).to(device)
    output = decode(model.generate(start_idx, max_new_tokens=max_new_tokens)[0].tolist())
    print(output)
    with open(f"outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(output)