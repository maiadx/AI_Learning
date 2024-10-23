import torch
import torch.nn as nn
from torch.nn import functional as Func


# hyperparameters:
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8   # what is the maximum context length for predictions?
max_iters = 3000 # training iters
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(2073)
print(torch.cuda.is_available())
print(device)
print(torch.version.cuda)
print(f"Available GPU mem: {torch.cuda.get_device_properties(0).total_memory}")

with open('gpt1/dataset/text.txt', encoding='utf-8') as fp:
    text = fp.read()


print(f"length of dataset in chars: {len(text)}\n")
#print(f"\"{text[:60]}\" ")

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(f"vocabulary: {''.join(vocab)}")
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
#input_str = "hii there"
#encoded_output = encode(input_str)
#print("Encoded:", encoded_output)

#decoded_output = decode(encoded_output)
#print("Decoded:", decoded_output)





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
# P(W)=P(w1​)⋅P(w2​∣w1​)⋅P(w3​∣w2​)⋅…⋅P(wn​∣wn−1​)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits of the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
            # idx and targets are both (B,T) tensor of integers
            logits = self.token_embedding_table(idx) # (B,T,C)
            # B, T, C: batch, time, channel

            if targets is None:
                loss = None
            else:
                # loss fn:
                # measures quality of logits w.r.t. the targets 
                # we have identity of next char, how well are we predicting next char based on logits?
                
                #pytorch expects different format for cross_entropy (B,C,T)
                # we need to reshape logits:
                B, T, C = logits.shape

                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                
                loss = Func.cross_entropy(logits, targets)

            return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context

        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)

             # focus only on the last time step:
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmas to get probabilities:
            probs = Func.softmax(logits, dim=-1) # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
model = BigramLanguageModel(vocab_size)
device_model = model.to(device)

# create pytorch optimizer:
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



# --------------
# Training loop:
# --------------

for steps in range(max_iters):

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(device_model.generate(context, max_new_tokens=500)[0].tolist()))

# mostly gibberish since our loss has plataued by this point at a high value.

# What could we do to improve upon this... 