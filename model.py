"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    ans = dict()
    cnt = len(specials);
    for idx, s in enumerate(specials):
        ans[s] = idx;


    for s in sentences:
        words = s.split()

        for w in words:
            if w not in ans:
                ans[w] = cnt
                cnt = cnt+1

    return ans

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    ans = dict()

    for k, v in token_to_id.items():
        ans[v] = k

    return ans

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    ans = []
    words = sentence.split()
    for w in words:
        if w in token_to_id:
            ans.append(token_to_id[w])
        else:
            ans.append(token_to_id[unk_token])

    return ans

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    ans = []
    for id_ in ids:
        ans.append(id_to_token[id_])

    return ans

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    length = len(ids)

    if length>=max_len:
        return ids[:max_len]

    else:
        while True:
            ids.append(pad_id)
            length = length+1
            if length==max_len:
                break

    return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return (torch.tensor)(padded_sequences)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings*math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    ans = []
    for i in range(d_model//2):
        ans.append(10000**((-2*i)/d_model))

    return (torch.tensor)(ans)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    ans = [[float(i)] for i in range(0, max_len)]
    return (torch.tensor)(ans)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    """Fill odd feature indices of pe with cos(position * div_term)."""
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch
import math

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""

    pe = torch.zeros(max_len, d_model)

    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    # Even indices: sine
    pe[:, 0::2] = torch.sin(position * div_term)

    # Odd indices: cosine
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    L = embedded_batch.shape[1]

    return embedded_batch + positional_encoding[:L, :]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        for j in range(seq_len):
            if(i>=j):
                mask[0][0][i][j] = True

    return mask

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    """Combine padding and causal masks using logical AND."""
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    return torch.matmul(query, key.transpose(-2, -1))

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(~mask.bool(), float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    """Compute attention weights, safely handling fully masked rows."""

    weights = torch.softmax(masked_scores, dim=-1)

    # Replace NaN values from fully masked rows with zero
    weights = torch.nan_to_num(weights, nan=0.0)

    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    context = torch.matmul(attention_weights, value)
    return context

# Step 22 - scaled_dot_product_attention
import torch
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""

    # Dimension of key vectors
    d_k = query.size(-1)

    # Step 1: Compute raw attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply attention mask
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float('-inf'))

    # Step 4: Compute attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Handle fully masked rows
    attention_weights = torch.where(
        torch.isneginf(scores).all(dim=-1, keepdim=True),
        torch.zeros_like(attention_weights),
        attention_weights
    )

    # Step 5: Compute weighted sum of values
    context = torch.matmul(attention_weights, value)

    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    B, L, d_model = tensor.shape
    tensor = tensor.reshape((B, L, num_heads, d_model//num_heads))
    return tensor

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B, num_heads, L, d_k = multi_head_tensor.shape
    return multi_head_tensor.transpose(1, 2).reshape(B, L, num_heads*d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is not None:
        return x@weight.T + bias
    
    else:
        return x@weight.T

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)

    return (q, k, v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q =  split_last_dim_into_heads(q, num_heads)
    k =  split_last_dim_into_heads(k, num_heads)
    v =  split_last_dim_into_heads(v, num_heads)

    q = transpose_heads_before_sequence(q)
    k = transpose_heads_before_sequence(k)
    v = transpose_heads_before_sequence(v)

    return (q, k, v)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    context, weights = scaled_dot_product_attention(q_h, k_h, v_h, mask)
    return (context, weights)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    context = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(context, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)

    q, k, v = split_qkv_into_heads(q, k, v, num_heads)

    context, atten_wt = scaled_dot_product_attention(q, k, v, mask)
    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    # Project x, add bias, and apply ReLU
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden@w2+b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
def compute_layer_norm_mean_and_variance(x):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # Compute mean and population variance
    mean, variance = compute_layer_norm_mean_and_variance(x)

    # Normalize
    x_norm = (x - mean) / torch.sqrt(variance + eps)

    # Apply learnable affine transformation
    return gamma * x_norm + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input+sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    """Apply inverted dropout using a precomputed boolean mask."""
    return x * keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    sublayer_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    sublayer_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    attn_sublayer_output = encoder_layer_self_attention_sublayer(x, layer_params['w_q'], layer_params['w_k'], layer_params['w_v'], layer_params['w_o'], layer_params['attn_gamma'], layer_params['attn_beta'], num_heads, src_mask)
    return encoder_layer_feed_forward_sublayer(attn_sublayer_output, layer_params['w1'], layer_params['b1'], layer_params['w2'], layer_params['b2'], layer_params['ffn_gamma'], layer_params['ffn_beta'])

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)

    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    sublayer_output = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.ndim == 2:
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    sublayer_output = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    sublayer_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""

    p = layer_params

    # 1. Masked self-attention
    y = decoder_layer_masked_self_attention_sublayer(
        y,
        p["w_q_self"],
        p["w_k_self"],
        p["w_v_self"],
        p["w_o_self"],
        p["self_gamma"],
        p["self_beta"],
        num_heads,
        tgt_mask
    )

    # 2. Cross-attention
    y = decoder_layer_cross_attention_sublayer(
        y,
        encoder_output,
        p["w_q_cross"],
        p["w_k_cross"],
        p["w_v_cross"],
        p["w_o_cross"],
        p["cross_gamma"],
        p["cross_beta"],
        num_heads,
        src_mask
    )

    # 3. Feed-forward network
    y = decoder_layer_feed_forward_sublayer(
        y,
        p["w1"],
        p["b1"],
        p["w2"],
        p["b2"],
        p["ffn_gamma"],
        p["ffn_beta"]
    )

    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)

    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    return token_embedding_weight.transpose(0, 1)

# Step 50 - apply_log_softmax_over_vocab
import torch

def apply_log_softmax_over_vocab(logits):
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
import torch

def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    p = model_params
    d_model = p["src_embedding"].shape[-1]

    # 1. Token embedding lookup
    x = p["src_embedding"][src_ids]
    y = p["tgt_embedding"][tgt_ids]

    # 2. Scale embeddings
    x = scale_embeddings_by_sqrt_d_model(x, d_model)
    y = scale_embeddings_by_sqrt_d_model(y, d_model)

    # 3. Build sinusoidal positional encoding
    max_len = max(src_ids.size(1), tgt_ids.size(1))
    pe = build_sinusoidal_positional_encoding(max_len, d_model)

    # 4. Add positional encoding
    x = add_positional_encoding_to_embeddings(x, pe.to(x))
    y = add_positional_encoding_to_embeddings(y, pe.to(y))

    # 5. Build source padding mask
    src_mask = build_padding_mask(src_ids, pad_id)

    # 6. Build combined target mask
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_ids.size(1))
    tgt_causal_mask = tgt_causal_mask.to(tgt_ids.device)

    tgt_mask = combine_padding_and_causal_masks(
        tgt_padding_mask, tgt_causal_mask
    )

    # 7. Run encoder stack
    encoder_output = stack_encoder_layers(
        x, p["encoder_layers"], num_heads, src_mask
    )

    # 8. Run decoder stack
    decoder_output = stack_decoder_layers(
        y, encoder_output, p["decoder_layers"],
        num_heads, src_mask, tgt_mask
    )

    # 9. Project decoder output to vocabulary logits
    logits = apply_final_output_projection(
        decoder_output, p["output_projection"]
    )

    # 10. Convert logits to log probabilities
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""

    def init_weight(fan_in, fan_out):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        return (
            torch.empty(fan_in, fan_out, dtype=torch.float32)
            .uniform_(-bound, bound)
            .requires_grad_(True)
        )

    def zeros(size):
        return torch.zeros(size, dtype=torch.float32, requires_grad=True)

    def ones(size):
        return torch.ones(size, dtype=torch.float32, requires_grad=True)

    return {
        # Self-attention projections
        "w_q": init_weight(d_model, d_model),
        "w_k": init_weight(d_model, d_model),
        "w_v": init_weight(d_model, d_model),
        "w_o": init_weight(d_model, d_model),

        # Feed-forward network
        "w1": init_weight(d_model, d_ff),
        "b1": zeros(d_ff),
        "w2": init_weight(d_ff, d_model),
        "b2": zeros(d_model),

        # Self-attention LayerNorm
        "attn_gamma": ones(d_model),
        "attn_beta": zeros(d_model),

        # Feed-forward LayerNorm
        "ffn_gamma": ones(d_model),
        "ffn_beta": zeros(d_model)
    }

# Step 53 - init_decoder_layer_parameters
import torch
import math

def init_decoder_layer_parameters(d_model, num_heads, d_ff):

    def init_weight(fan_in, fan_out):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        return (
            torch.empty(fan_in, fan_out, dtype=torch.float32)
            .uniform_(-bound, bound)
            .requires_grad_(True)
        )

    def zeros(size):
        return torch.zeros(size, dtype=torch.float32, requires_grad=True)

    def ones(size):
        return torch.ones(size, dtype=torch.float32, requires_grad=True)

    return {
        # Masked self-attention
        "w_q_self": init_weight(d_model, d_model),
        "w_k_self": init_weight(d_model, d_model),
        "w_v_self": init_weight(d_model, d_model),
        "w_o_self": init_weight(d_model, d_model),
        "self_gamma": ones(d_model),
        "self_beta": zeros(d_model),

        # Cross-attention
        "w_q_cross": init_weight(d_model, d_model),
        "w_k_cross": init_weight(d_model, d_model),
        "w_v_cross": init_weight(d_model, d_model),
        "w_o_cross": init_weight(d_model, d_model),
        "cross_gamma": ones(d_model),
        "cross_beta": zeros(d_model),

        # Feed-forward network
        "w1": init_weight(d_model, d_ff),
        "b1": zeros(d_ff),
        "w2": init_weight(d_ff, d_model),
        "b2": zeros(d_model),
        "ffn_gamma": ones(d_model),
        "ffn_beta": zeros(d_model)
    }

# Step 54 - init_embedding_and_projection_parameters
import torch
import math

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):

    def init_weight():
        bound = math.sqrt(6.0 / (vocab_size + d_model))
        return (
            torch.empty(vocab_size, d_model, dtype=torch.float32)
            .uniform_(-bound, bound)
            .requires_grad_(True)
        )

    src_embedding = init_weight()
    tgt_embedding = init_weight()

    output_projection = tgt_embedding if tie_weights else init_weight()

    return {
        "src_embedding": src_embedding,
        "tgt_embedding": tgt_embedding,
        "output_projection": output_projection
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(
    encoder_layer_params, decoder_layer_params, embedding_params
):
    model_params_list = []
    seen = set()

    def add_parameters(params):
        for tensor in params.values():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                model_params_list.append(tensor)

    for layer in encoder_layer_params:
        add_parameters(layer)

    for layer in decoder_layer_params:
        add_parameters(layer)

    add_parameters(embedding_params)

    return model_params_list

# Step 56 - shift_targets_right_with_start_token
import torch

def shift_targets_right_with_start_token(target_ids, start_token_id):
    batch, tgt_seq = target_ids.shape

    shifted_tgt_seqs = [[0 for _ in range(tgt_seq)] for _ in range(batch)]

    for i in range(batch):
        shifted_tgt_seqs[i][0] = start_token_id

        for j in range(1, tgt_seq):
            shifted_tgt_seqs[i][j] = target_ids[i][j - 1].item()

    return torch.tensor(
        shifted_tgt_seqs,
        dtype=target_ids.dtype,
        device=target_ids.device
    )

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    if step == 0:
        return 0.0

    return float(
        d_model ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    ans = torch.ones(shape)*(epsilon/(vocab_size-2))
    return ans

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""

    return smoothed_distribution.scatter(
        dim=-1,
        index=gold_token_ids.unsqueeze(-1),
        value=confidence
    )

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 68 - compute_adam_parameter_update (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

