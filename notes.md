- Batch Normalization: Normalizes each feature across all the batches (Vertical) by a normal dist. (mean = 0, std_dev = 1)
- Layer Normalization: across the feature space (Horizontal)

- Feed forward NN typically implies involvement of activation layers

- We use Add and Norm

### Logits 
Raw unnormalized scores that come out of the last layer of a NN before any softmax or activation.

# Transformer

## Attention
 - Each pair of tokens is compared(Q, K each of dimension dk), their dot product is divided by sq. root dk, softmax is taken to normalize and calculate weights for the values. Then the output is weights @ V.

## Masking

### Casual Masking
Prevents tokens from seeing future tokens during training.
- uses torch.tril()
- scores.masked_fill(mask == 0, float('-inf)). masked_fill because we want 0 probab through -inf in softmax.

### Padding Mask
Handles variable length sequences in a batch by padding shorter sequences with 0s

## Postitional Encoding
 Base transformer models use sinuosidal positional encodings, whereas GPTs use learnable embeddings
