# First check 01-shakespeare.py since it introduces the Transformer model.

# https://github.com/openai/gpt-2
# Paper: Language Models are Unsupervised Multitask Learners
# Paper: Language Models are Few-Shot Learners
# Paper: Attention is all you need

# GPT-2 was written with Tensor Flow, but Huggingface has its implementation in PyTorch
# https://huggingface.co/openai-community/gpt2
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

from transformers import GPT2LMHeadModel, pipeline, set_seed
import matplotlib.pyplot as plt

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