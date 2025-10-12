# Check stats.py first.
# Same problem, but we'll use a neural network instead of just statistics

import torch, torch.nn.functional as F, matplotlib.pyplot as plt
from mmshared import *

# random generator
g = torch.Generator().manual_seed(2147483647)

words = get_words()
stoi, itos = get_dictionaries(words)

# Let's create a training set of bigrams (x, y). Meaning: x -> y.
# We'll use vectors xs and ys.
# xs is array of inputs.
# ys is array of outputs.
# (but indexes, not chars)
# Example: in case of .emma.:
# xs: . e m m a
# xy: e m m a .

def create_training_set(words):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    return torch.tensor(xs), torch.tensor(ys)

xs, ys = create_training_set(words[:1])

num = xs.nelement()
print(f"number of examples: ", num)

# Initialize the neural network

# We have 27 letters, each letter is represented not by a number, but with one-hot vector, i.e. [0,..,1,..,0] where 1 is on the position of that number.
# Neural network input should be one-hot vector since internally it multiplies numbers.

xenc = F.one_hot(xs, num_classes = 27).float()

# xenc is multiple chars (inputs) encoded to one-hot.
# It is a matrix where each row is one-hot vector of appropriate input example (number).
# Neural net works with multiple input examples simultaneously packed into a matrix.

print(xenc.shape)
plt.imshow(xenc)
# plt.show()


# define initial random weights
W = torch.randn((27, 27), generator = g, requires_grad = True)

# Gradient descent

for k in range(100):

  # Forward pass

  # Our layer has no bias and no activation function, so we just multiply input by weights.
  # Similarly to stats.py, we want output to be an analogue of counts and then we can calculate probabilities.
  # Neural network does not work with int counts as they may be large, so we consider it will output "log counts" called "logits".
  # If we exponentiate them using exp() then we get equivalent of counts (matrix N).

  logits = xenc @ W
  counts = logits.exp()

  # Similarly how we calculated P, we calc probabilities here.
  # Every row is a row of probabilities of one of the 27 letters.
  # I.e. you may see it as like every input one-hot vector represent 100% probability (1 is 100%) of specific letter,
  # and now we have non-100% for every letter.
  # Such exp() + normalizing is called "soft-max function for logits".
  # We convert [1.3, 5.1, 2.2, 0.7, 1.1] -- softmax -> [0.02, 0.9, 0.05, 0.01, 0.02]. Higher the number - higher the prob, sum is 1.

  probs = counts / counts.sum(1, keepdim = True)

  # Btw, remember that we have multiple examples in the matrix, each row is one example.

  # Let's try to tune our W to improve probs by reducing our loss function (defined in stats.py).

  # Calculate loss (average negative log likelyhood)

  # for-loop form
  # nlls = torch.zeros(num)
  # for i in range(num):
  #   x = xs[i].item()
  #   y = ys[i].item()
  #   p = probs[i, y] # probs[i] is probabilities for all chars, so probs[i, y] is probability assigned to y
  #   nll = -torch.log(p)
  #   nlls[i] = nll
  # loss = nlls.mean().item()

  # tensor form
  loss = -probs[torch.arange(num), ys].log().mean()
  
  # Example:
  # torch.arange(5) = [0, 1, 2, 3, 4]

  print(loss.item())

  # backward pass

  W.grad = None # set gradient to 0
  loss.backward()

  # update weights

  W.data += -50 * W.grad