import torch
import torch.nn as nn
from torch.nn import functional as Fn


# hyperparameters:
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256   # what is the maximum context length for predictions?
max_iters = 5000 # training iters
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed_dim = 384 # dimension table for 32 dimensional embeddings
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(2073)

#print(torch.cuda.is_available())
print(f"{device} v{torch.version.cuda}")
print(f"Available GPU mem: {torch.cuda.get_device_properties(0).total_memory}")

with open('gpt1/data/text.txt', encoding='utf-8') as fp:
    text = fp.read()


print(f"length of dataset in chars: {len(text)}\n")
#print(f"\"{text[:60]}\" ")

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
#print(f"vocabulary: {''.join(vocab)}")
print(f"vocab size: {vocab_size}")


# ------------------------------------------------------
# strategy to tokensize text -> model is character-level,
#  so we need to translate characters into integers.

# encode - string to integers (list)
def encode(input_string):
    # Define the vocabulary    
    # Create a dictionary to map each character to an integer
    char_to_index = {char: idx for idx, char in enumerate(vocab)}
    
    # Convert input string to an array of integers based on the character mapping
    encoded_array = list(map(lambda char: char_to_index[char], input_string))    
    return encoded_array


def decode(encoded_data):
    
    # mapping for integers back into string:
    index_to_char = {idx: char for idx, char in enumerate(vocab)}

    # convert the array of integers back to a string based on the reverse mapping:
    decoded_string = ''.join(map(lambda idx: index_to_char.get(idx, ''), encoded_data))
    return decoded_string


# Example usage of encode and decode
# input_str = "hii there"
# encoded_output = encode(input_str)
# print("Encoded:", encoded_output)

# decoded_output = decode(encoded_output)
# print("Decoded:", decoded_output)



# ------------------
# test / train split
# ------------------

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]



# ------------------
#   Data loading
# ------------------

def get_batch(split):
    # gen a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    
    return x, y


@torch.no_grad()
def estimate_loss():
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


# bigram language model is the simplest model for nlp.
# bigram predicts next word solely based on the preceeding word. (n-gram where n = 2)
# P(W) = P(w1​) ⋅ P(w2 ​∣ w1​) ⋅ P(w3 ​∣ w2​)⋅ … ⋅ P(wn ​∣ wn−1​)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed_dim, head_size, bias=False)
        self.query = nn.Linear(n_embed_dim, head_size, bias=False)
        self.value = nn.Linear(n_embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = Fn.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed_dim, 4 * n_embed_dim),
            nn.ReLU(),
            nn.Linear(4 * n_embed_dim, n_embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computate """

    def __init__(self, n_embed_dim, n_head):
        # n_embed_dim: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed_dim // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed_dim)
        self.ln1 = nn.LayerNorm(n_embed_dim)
        self.ln2 = nn.LayerNorm(n_embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, n_embed_dim)
        self.blocks = nn.Sequential(*[Block(n_embed_dim, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed_dim) # final layer norm
        self.lm_head = nn.Linear(n_embed_dim, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = Fn.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = Fn.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M paramters')

# create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(m.generate(context, max_new_tokens=500)[0].tolist())
open('gpt1/gpt1-output.txt', 'w').write(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
print("output to file")

# -----------------------------------------------------------------------------------------------------
