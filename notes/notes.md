# Language model
The model generates text autoregressively, predicting one token at a time based on all previous tokens. The causal attention mask ensures that generation maintains the same constraints as training, where each prediction only depends on past context.

## Architecture
The language model chosen is a generative pretrained transformer similar to GPT-2 architecture. It uses only a decoder (i.e. masking future tokens for self attention) and consists of

### Multi-head Attention
Multi-head attention allows the model to attend to different representation subspaces simultaneously. The attention mechanism is split into multiple "heads", each learning different types of relationships between tokens. Each head has its own query $Q$, key $K$, and value $V$ projection matrices, allowing the model to capture various linguistic patterns like syntactic dependencies, semantic relationships, and positional associations in parallel.

The multi-head mechanism works by:
1. Projecting input embeddings into multiple $Q$, $K$, $V$ linear spaces
2. Computing scaled dot-product attention for each head independently  
3. Concatenating all head outputs
4. Applying a final linear projection to combine the information

### Causal Self-Attention
Causal self-attention is used in the decoder block of a transformer. It attends only to previous tokens in the sequence using a lower triangular mask. This ensures that predictions for position $i$ can only depend on tokens at positions less than $i$, enabling autoregressive training where the model learns to predict the next token given all previous tokens. The causal mask prevents information leakage from future tokens during training.

### Positional Embeddings
Since transformers have no inherent notion of sequence order, positional embeddings are added to input token embeddings to encode positional information. The model uses learned positional embeddings rather than sinusoidal encodings, where each position (`0 to max_seq_len-1`) has its own trainable embedding vector. These positional embeddings are added element-wise to token embeddings before being fed into the transformer blocks.

### Weight Sharing
The model implements weight tying between the input token embedding matrix (`token_emb.weight`) and the output language modeling head (`lm_head.weight`). This reduces the number of parameters and often improves performance by ensuring that tokens with similar meanings have consistent representations in both input and output spaces. This is a common practice in language models that helps with training stability and parameter efficiency.

### Pre-layer Normalization
The model uses pre-layer normalization (Pre-LN) rather than post-layer normalization. In each transformer block, layer normalization is applied before the attention and MLP sublayers.

The residual connections add the normalized sublayer output to the original input: `x = x + sublayer(LayerNorm(x))`

## Training

### Dataset
The model is trained on the TinyStories dataset, a collection of short stories written in simple language suitable for children. The dataset contains synthetic stories generated to help train small language models while maintaining narrative coherence and educational value.

### Training Procedure
The model is trained using teacher forcing, where the input sequence is the story text and the target is the same sequence shifted by one position. Cross-entropy loss is computed between predicted logits and target tokens. The training uses:
- Autoregressive next-token prediction objective
- Sequence length of 256 tokens maximum
- GPT-2 tokenizer for consistent vocabulary

### Compilation
PyTorch 2.0's `torch.compile()` is used to optimize the model for faster training through graph compilation and kernel fusion, though care must be taken when saving/loading checkpoints due to the `_orig_mod` wrapper.

## Text Generation

### Sampling Strategies
The model does next token prediction by drawing it from a multinomial distribution if temperature scaling is used. Otherwise it selectes the most likely token. The class `Inference` supports multiple text generation approaches:

**Temperature Sampling**: Controls randomness by scaling logits before applying softmax. Higher temperatures ($>1.0$) increase diversity, lower temperatures ($<1.0$) make outputs more focused and deterministic. For temperature $0.0$ the inference is deterministic.

**Top-k Sampling**: Only considers the k most likely next tokens in the multinomial distribution, setting all other probabilities to zero. This prevents sampling from very unlikely tokens while maintaining some randomness.

**Top-p (Nucleus) Sampling**: Dynamically selects the smallest set of tokens whose cumulative probability exceeds threshold p. This adapts the candidate set size based on the prediction confidence.


### Generation Process
1. Encode the input prompt using the tokenizer
2. Feed tokens through the model to get next-token logits
3. Apply sampling strategy to select next token
4. Append selected token to sequence
5. Repeat until end-of-sequence token or maximum length reached
6. Decode final token sequence back to text