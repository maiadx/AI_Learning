# following the Andrej Karpathy Reproducing GPT-2 video 


from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as Fn


# -------------------
#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{device} v{torch.version.cuda}")
print(f"Available GPU mem: {torch.cuda.get_device_properties(0).total_memory}")

with open('gpt2/dataset/text.txt', encoding='utf-8') as fp:
    text = fp.read()




@dataclass
class GPTConfig: 
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    class GPT(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.config = config

            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed),
                wpe = nn.Embedding(config.block_size, config.n_embed),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embed),
            ))

            self.lm_head = nn.Linear(config.n_embed, config/vocab_size, bias=False)
