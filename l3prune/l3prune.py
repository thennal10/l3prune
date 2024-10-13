import random
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from .model_overrides import get_forward

# A custom encode function to override the forward of the model
def encode_custom(forward, encoder, sentence_feature):
    embed_mask = None
    if "embed_mask" in sentence_feature:
        embed_mask = sentence_feature.pop("embed_mask")
    out, reps = forward(encoder.model, **sentence_feature)
    sentence_feature["embed_mask"] = embed_mask

    return [encoder.get_pooling(sentence_feature, emb) for emb in reps]

def l3prune(encoder, dataset, loss_fn, batch_size=64, num_samples=100):
    dataset = [t for t in dataset]
    subset = random.sample(dataset, batch_size*num_samples)
    subset = [[encoder.prepare_for_tokenization(t) for t in s.texts] for s in subset]
    subset = [subset[i:i + batch_size] for i in range(0, len(subset), batch_size)]

    num_layers = encoder.model.config.num_hidden_layers
    loss = {i: [] for i in range(1, num_layers+1)}
    forward = get_forward(encoder.model)

    with torch.no_grad():
        # Override the forward of the model to get the intermediate representations in only one pass
        if forward:
            encode = partial(encode_custom, forward)
            for batch in tqdm(subset):
                features = []
                for j in range(3):
                    embs = [t[j] for t in batch]
                    embs = encoder.tokenize(embs).to(encoder.model.device)
                    embs = encode(encoder, embs)
                    features += [embs]
                q, d, d_neg = features
                for i in range(num_layers):
                    loss[i+1] += [loss_fn(q[i], d[i], d_neg[i])]
        else:
            # Without the override, we have to rerun the forward pass with each layer pruned
            for l in range(num_layers, 0, -1):
                encoder.prune(layer_prune=l)
                for batch in tqdm(subset):
                    features = []
                    for j in range(3):
                        embs = [t[j] for t in batch]
                        embs = encoder.tokenize(embs).to(encoder.model.device)
                        embs = encoder.forward(embs)
                        features += [embs]
                    q, d, d_neg = features
                    loss[l] += [loss_fn(q, d, d_neg)]

        loss = [torch.tensor(loss[i]).mean().float().detach() for i in range(1, num_layers+1)]
    
    # minima before and after midpoint
    midpoint = num_layers // 2
    small_p = np.argmin(loss[:midpoint]) + 1
    large_p = np.argmin(loss[midpoint:]) + midpoint + 1
    return small_p, large_p