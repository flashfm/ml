# First check 01-shakespeare.py since it introduces the Transformer model.

# https://github.com/openai/gpt-2
# Paper: Language Models are Unsupervised Multitask Learners
# Paper: Language Models are Few-Shot Learners
# Paper: Attention is all you need

# GPT-2 was written with Tensor Flow, but Huggingface has its implementation in PyTorch
# https://huggingface.co/openai-community/gpt2
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

from transformers import GPT2LMHeadModel, pipeline, set_seed
from dataclasses import dataclass
from torch.nn import functional as F
import matplotlib.pyplot as plt, torch, torch.nn as nn, tiktoken, math

# First let's try HuggingFace implementation and weights of actual GPT-2 and generate some text

def try_huggingface():
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # gpt2 is 124 M parameters, gpt2-xl is 1.5 B parameters
    sd_hf = model_hf.state_dict()

    # Prints all layers and weight shapes
    for k,v in sd_hf.items():
        print(k, v.shape)

    # transformer.wte.weight torch.Size([50257, 768]) - means 50257 tokens in vocabulary, 768 is token embedding size
    # transformer.wpe.weight torch.Size([1024, 768]) - means context size is 1024, 768 is position embedding size

    #position_embedding = sd_hf["transformer.wpe.weight"]
    #plt.imshow(position_embedding, cmap="gray") # Positional embedding shows some structure

    # Actual features look like sin/cos charts.

    #plt.plot(position_embedding[:, 150])
    #plt.plot(position_embedding[:, 200])
    #plt.plot(position_embedding[:, 250])
    #plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300,:300], cmap="gray")
    #plt.show()

    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    g = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5, truncation=True)
    print(g)

# Let's now reproduce GPT-2 ourselves.
# GPT-2 does not have encoder part and cross-attention part in the decoder block.
# Also, in the GPT-2 paper they tell that thay changed LayerNorm layers positions comparing to original Attention paper.

# Looking at the layer names in the sd_hf dictionary, reproducing the layers:

@dataclass
class GPTConfig:
    block_size: int = 1024 # context length
    vocab_size: int = 50257 # 50K BPE merges, 256 byte tokens + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections, in all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask, but in OpenAI/HF naming called "bias"
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        # qkv is for all heads and in batch
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        # nh = number of heads
        # hs = head size
        # C = nh * hs
        # In GPT-2 (124M), n_head = 12, hs = 64, C = 12*64 = 768 channels
        # This is equivalent to MultiHeadAttention from 01-shakespeare, but more effective since (B, nh) is treated as batch dimension and processed in parallel

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # Paper: Gaussian Error Linear Units (GELUs)
        # The approximate version: https://github.com/pytorch/pytorch/issues/39853
        # It is not really needed now, but was used since the original version was slow in TensorFlow
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # Forward is needed for generation from the model
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # positions, (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)
        # Classifier (Linear)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("Loading weights of %s" % model_type)
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),  # 124 M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350 M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280), # 774 M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600), # 1558 M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # ignore the mask

        # load Huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # prepare keys (some weights are transposed)
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys if not k.endswith(".attn.masked_bias")] # ignore the mask
        sd_keys_hf = [k for k in sd_keys if not k.endswith(".attn.bias")] # ignore the mask

        # some weights in TensorFlow are transposed, but we want it back normal
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys)

        # copy weights
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

model = GPT.from_pretrained("gpt2")

num_return_sequences = 5
max_length = 30

# Switch to evaluation mode.
# We don't have Batch Norm or others which differ in train/eval mode, but anyway.
model.eval()

# Switch to CUDA.
#device = "cuda"
device = "cpu"
model.to(device)

# Let's generate!
# We use Tiktoken tokenizer to get tokens from string and back.

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,") # 8 integers
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:,-1,:] # last (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # Top-k sampling of 50 (Huggingface default)
        # This means that we sort probs and everything over 50th is replaced to 0, then normalized again
        # This way we have no chance to sample very rare tokens.
        # topk_probs and topk_indices are (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # ?
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1)

# print
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)