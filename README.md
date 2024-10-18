
# L3Prune: Large Language Models Are Overparameterized Text Encoders


L3Prune is a pruning method for LLM-based text encoders. Based simple layer-dropping and supervised fine-tuning, L3Prune can reduce the number of parameters of an LLM-based text encoder by 30% with negligible performance loss and up to 80% while maintaining reasonable performance. Much of this codebase is adapted from the [LLM2Vec repository](https://github.com/McGill-NLP/llm2vec).

## Overview

Given a model, its config, and a pruning percentage `p`, simple layer-dropping can be implemented in just three lines: 
```python
n = int(config.num_hidden_layers * (1-p))
model.layers = model.layers[:n]
config.num_hidden_layers = n
```

Supervised finetuning (generally paramater-efficient) is considered the most effective strategy to convert LLMs to effective text encoders. By applying said supervised finetuning after pruning, the lost performance is recovered. Thus, this method is easily integrated into any LLM-to-text-encoder pipeline, and can be applied without additional computation.

L3Prune goes further, and uses the initial layerwise loss of the base model as a guideline to pick layers to prune to. Two pruning configurations, `large` and `small` are produced by L3Prune, usable in different circumstances. `large` in particular has a negligible performance drop, while shaving off 21% of the model's parameters on average. Refer to the paper for more details.


## Installation
To use L3Prune, clone the repository and install the requirements.
```bash
git clone https://github.com/thennal10/l3prune.git
cd l3prune
pip install -r requirements.txt
```

## Getting Started
The LLMEncoder class is a wrapper around the HuggingFace transformers library, and can be used to encode text. It can be directly pruned using the `prune` method by passing the desired pruning percentage.

### Preparing the model
The `from_pretrained` method of an LLMEncoder takes a base model identifier/path. All HuggingFace model loading arguments can be passed to `from_pretrained` method. The `pooling_mode` argument can be used to change the pooling strategy, and the `max_length` argument can be used to change the maximum sequence length.

```python
import torch
from l3prune import LLMEncoder

encoder = LLMEncoder.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    pooling_mode="weighted_mean",
    torch_dtype=torch.bfloat16,
)
```

### Basic Pruning

Simple layer-dropping can be done by calling the `prune` method of the encoder, which takes the desired pruning percentage as an argument. If the pruning percentage is greater than or equal to 1, it is instead taken as the specific layer number to prune to.

```python
encoder.prune(0.3) # Prune 30% of the model
# OR
encoder.prune(8) # Prune to the 8th layer (if the model had 32 layers, this would be equivalent to p=0.75)
```


### Inference
This model now returns the text embedding for any input in the form of `[[instruction1, text1], [instruction2, text2]]` or `[text1, text2]`. While training, we provide instructions for both sentences in symmetric tasks, and only for for queries in asymmetric tasks.

```python
# Encoding queries using instructions
instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "how much protein should a female eat"],
    [instruction, "summit define"],
]
q_reps = encoder.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = encoder.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.6470, 0.1619],
        [0.0786, 0.5844]])
"""
```

## L3Prune

The `l3prune` function applies L3Prune to a given LLMEncoder and outputs two configurations, `large` and `small`. The `large` configuration is designed to have a negligible performance drop, while the `small` configuration is designed to be as small as possible while maintaining reasonable performance. The `l3prune` function takes the following arguments:

- `encoder`: The LLMEncoder to prune.
- `dataset`: The dataset to use for calculating the initial layerwise loss.
- `loss_fn`: The loss function.
- `batch_size`: The batch size.
- `num_batches`: The number of batches to take from the dataset.

Here is an example of how to use the `l3prune` function:

```python
import torch
from l3prune import LLMEncoder, l3prune
from l3prune.dataset.utils import load_dataset
from l3prune.loss.utils import load_loss
from accelerate import Accelerator
from accelerate.logging import get_logger

accelerator = Accelerator() # required for the logger in load_dataset

dataset = load_dataset(
        "E5",
        split="train",
        file_path="cache/echo-data",
        effective_batch_size=64,
    )
loss_fn = load_loss("HardNegativeNLLLoss")

encoder = LLMEncoder.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="cuda",
    pooling_mode="weighted_mean",
    torch_dtype=torch.float16,
)

small_p, large_p = l3prune(encoder, dataset, loss_fn)
```

`small_p` and `large_p` are the pruning layers for the `small` and `large` configurations, respectively. These values can be used to prune the model using the `prune` method of the encoder.

## Training 

We use the public portion of dataset used in [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368), curated by authors of [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449). The full description of the dataset can be found in Appendix A of our paper as well. The dataset can be downloaded from the [GitHub page of Echo embeddings repository](https://github.com/jakespringer/echo-embeddings#training). To use the training script, the downloaded dataset should be placed in the `cache` directory. The directory layout should be as follows:

```
cache
|── wiki1m_for_simcse.txt
└── echo-data
    ├── allnli_split1.jsonl
    ├── allnli_split2.jsonl
    ├── allnli.jsonl
    ├── dureader.jsonl
    ...
```
If the dataset is placed in a different directory, please change the `dataset_file_path` in the training configuration accordingly. 

To train the LLaMA-3-8B model with supervised contrastive learning, run the following command:

```bash
torchrun --nproc_per_node=8 train.py configs/MetaLlama3.json
```
The number of GPUs can be changed by modifying the `--nproc_per_node` argument. Alternatively, for a single GPU system, you can simply do:
```
python train.py configs/MetaLlama3.json
```

### Configuratioin
The training configuration files in the [config](config) directory defines the hyperparameters of the training runs, along with the  pruning. For example, here's the config file for LLaMA-3-8B: 

```json
{
    "model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
    "pooling_mode": "weighted_mean",
    "dataset_name": "E5",
    "dataset_file_path": "cache/echo-data",
    "remove_unused_columns": false,
    "learning_rate": 0.0002,
    "num_train_epochs": 3,
    "warmup_steps": 300,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "do_train": true,
    "disable_tqdm": false,
    "max_seq_length": 512,
    "overwrite_output_dir": true,
    "output_dir": "output/meta-llama/Meta-Llama-3-8B-Instruct",
    "use_adapter": true,
    "percent_prune": [25],
    "autoprune": "small+large",    
    // ....
}
```

Of particular note is the `percent_prune` and `autoprune` configurations. `percent_prune` provides a list of pruning values (either less than 1 as a fraction to prune aaway, or greater than 1 as the layer to prune to). Seperate models will be trained for each value in the `percent_prune` list. `autoprune`, if set, will automatically prune the model via L3Prune. By default, `autoprune` will override `percent_prune` unless it is set to `all`. The options included:
- `small`: Apply L3Prune, and prune and train only the `small` configuration.
- `large`: Apply L3Prune, and prune and train only the `large` configuration.
- `small+large`: Apply L3Prune, and prune and train both the `small` and `large` configurations.
- `all`: Apply L3Prune, and prune and train all configurations in `percent_prune`, as well as the `small` and `large` configurations.
