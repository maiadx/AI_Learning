
Started with ngram model for simplicity, move to transformers to get much better quality output. 


Important hyperparameters: 
	- batch size -> how many independent sequences of text will we process concurrently?
	- block size -> what is the maximum context length for predictions?
	- max iters -> training iterations
	- learning rate -> self expl.
	- n embed dim -> (384) dimension table for 32 dimensional embeddings 
			384/32 is 12, need to revisit why its this value in video

Encoder -> create a dictionary to map each token to an integer value for our model to learn. Uses map to index.

Decoder -> inversely map that value back to the text

----------------------------------------------------------------
NGram Model : 

----------------------------------------------------------
**Attention Head** : *Attention is the most important component of GPT1*

*init() * 
```init(head_size)
	self.key = nn.Linear(n_embed_dim, head_size, bias=False)
	self.query = nn.Linear(n_embed_dim, head_size, bias=False)
	self.value = nn.Linear(n_embed_dim, head_size, bias=False)
```

*forward() * 
```forward(x)
	input : (batch, time-step, channels)
	output : (batch, time-step, head size)
	
	key, query are (Batch, Time, head_size)
	
	weights = q @ transpose(-2,-1) * k.shape[-1]**-0.5
	
	weights = weights.masked_fill(self.tril[:T, :T] == 0, '-inf')
	# ^ all weights that are zero get set to negative infinity
	
	weights = softmax(weights) # activation fn
	weights = dropout(weights) 
	# perform aggregation of the values
	v = value(x)        # (B, T, hs)
	out = weights @ v   # (B, T, T) @ (B, T, hs) -> (B, T, hs)
	return out
```


#### **Multi-head Attention :**

*init(num_heads, head_size)*
```init(num_heads, head_size)
	heads = nn.ModuleList([AttnHead(head_size) for _ in range(num_heads)])
	proj = nn.Linear(head_size * num_heads, n_embed_dim)
	dropout = Dropout(dropout)
```

*forward(x) * 
```forward
	out = torch.cat([h(x) for h in self.heads], dim=-1)
	out = self.dropout(self.proj(out))
	return out
```


#### **FeedForwardNN  :* a simple layer followed by a non-linearity*

*init(n_embed_dim) *
```init
	net = Sequential(
				Linear(n_embed_dim, 4 * n_embed_dim),
				ReLu(),
				Linear(4 * n_embed_dim, n_embed_dim),
				Dropout(dropout)
	)
```

*forward(x) :*
```return net(x)```


#### **Block** : *Transformer Block -> communication followed by computation*

*init(n_embed_dim, n_head) *
```init
	head_size = n_embed_dim // n_head
	sa = MultiHeadAttention(n_head, head_size)
	ffwd = FeedForward(n_embed_dim)
	ln1 = LayerNorm(n_embed_dim)
	ln2 = LayerNorm(n_embed_dim) 
```

*forward(x)*
```forward
	x = x + sa(ln1(x))
	x = x + ffw(self.ln2(x))
	return x
```


#### **GPTLanguageModel :** 

init() 
```init
	# each token directly reads off the logits for the next token from a lookup 
																			table
	token_embedding_table = nn.Embedding(vocab_size, n_embed_dim)
	position_embedding_table = nn.Embedding(block_size, n_embed_dim)

	blocks = nn.Sequential(*[Block(n_embed_dim, n_head=n_head) for _ in 
															range(n_layer)])
	ln_f = nn.LayerNorm(n_embed_dim) # final layer norm
	lm_head = nn.Linear(n_embed_dim, vocab_size)
	# better init, not covered in the original GPT video, but important, will 
														cover in followup video
	self.apply(self._init_weights)
```

init_weights() 
```init_weights
	if isinstance(module, nn.Linear):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
	
	if module.bias is not None:
		torch.nn.init.zeros_(module.bias)
	
	elif isinstance(module, nn.Embedding):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

forward(idx, targets=None)
```forward
B, T = idx.shape
# idx and targets are both (B,T) tensor of integers
tok_emb = self.token_embedding_table(idx) # (B,T,C)
pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

x = tok_emb + pos_emb # (B,T,C)

x = self.blocks(x) # (B,T,C)

x = self.ln_f(x) # (B,T,C)

logits = self.lm_head(x) # (B,T,vocab_size)
```

generate(idx, max_new_tokens) 
```generate
for _ in range(max_new_tokens):
	# get the predictions
    logits, loss = self(idx_cond)
    
    # focus only on the last time step
    logits = logits[:, -1, :] # becomes (B, C)
    # apply softmax to get probabilities
    probs = Fn.softmax(logits, dim=-1) # (B, C)
    
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1) 
												# ^ (B, 1)
    # append sampled index to the running sequence
    idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
```

________________________________________________________________________

## Running the Code : 

```
----- Training -----
for iter in range (max_iters): 
	if iter % eval_interval == 0 or iter == max_iters - 1:
		losses = estimate_loss()
		# also print loss

	xb, yb = get_batch('train')

	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=T rue)
	loss.backward()
	optimizer.step()
	
	# sample a batch of data
	xb, yb = get_batch('train')

	# evaluate the loss 
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
```

```
----- Test the trained model -----
context = torch.zeros((1, 1), dtype=torch.long, device=device)

output = decode(m.generate(context, max_new_tokens=500)[0].tolist())

# write some output to a file that we can view : 
open('gpt1/gpt1-output.txt', 'w').write(decode(m.generate(context, 
											max_new_tokens=1000)[0].tolist()))
```

