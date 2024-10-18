import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from l3prune import LLMEncoder, l3prune
from l3prune.dataset.utils import load_dataset
from l3prune.loss.utils import load_loss

from tqdm import tqdm

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
        "Phi3Config"
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )
    percent_prune: Optional[tuple] = field(
        default=(0,),
        metadata={"help": """A list of percentages to prune away. If greater than 1, it is taken as the layer nunber to prune to.
                  e.g. if it is 0.3, 30%% of the model will be pruned, if it is 10, the model will be pruned to 10 layers."""}
    )
    autoprune : Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Automatically prune the model based on the loss calculations, i.e L3Prune. Overrides percent_prune unless set to all, "
                "in which case small and large are added to the list of percentages to prune away."
            ),
            "choices": ["small", "large", "small+large", "all"]
        }
    )
    use_adapter: Optional[bool] = field(
        default=True,
        metadata={"help": "Use LoRA adapter."}
    )
    save_full_model: Optional[bool] = field(
        default=False,
        metadata={"help": """Save the full (pruned) model after training. If false, only the adapters will be saved, 
                    requiring the base model to be loaded in full before being pruned when used for inference."""}
    )  


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )


@dataclass
class DefaultCollator:
    model: LLMEncoder

    def __init__(self, model: LLMEncoder) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(
                    self.model, text, pooling_mode=self.model.pooling_mode
                )
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class LLMEncoderSupervisedTrainer(Trainer):

    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        loss = self.loss_function(q_reps, d_reps, d_reps_neg)

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def get_train_dataloader(self) -> DataLoader:
        # Copying most of the code from the parent class, changing the sampler to SequentialSampler
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # Changing from random sampler to sequential sampler
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # TODO: can also pass separator arg here
    train_dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
        effective_batch_size=training_args.per_device_train_batch_size
        * accelerator.num_processes,
    )

    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]


    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLMEncoder.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        device_map="cuda"
    )
    
    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    # L3Prune 
    if model_args.autoprune:
        small_p, large_p = l3prune(model, train_examples, train_loss, num_samples=2)
        if model_args.autoprune == "small":
            prunes = (small_p,)
        elif model_args.autoprune == "large":
            prunes = (large_p,)
        elif model_args.autoprune == "small+large":
            prunes = (small_p, large_p)
        elif model_args.autoprune == "all":
            prunes = tuple(model_args.percent_prune) + (small_p, large_p)
        else:
            raise ValueError("Invalid autoprune value")
    else:
        prunes = model_args.percent_prune
    
    logger.info(f"Pruning configurations: {prunes}")
    output_dir = training_args.output_dir
    for p in prunes:
        if custom_args.experiment_id is not None:
            experiment_id = custom_args.experiment_id
        else:
            experiment_id = p

        training_args.output_dir = f"{output_dir}/{experiment_id}"

        # reset model
        model = LLMEncoder.from_pretrained(
            base_model_name_or_path=model_args.model_name_or_path,
            pooling_mode=model_args.pooling_mode,
            max_length=model_args.max_seq_length,
            torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
            device_map="cuda"
        )
        model.prune(p)
        if model_args.use_adapter:
            # model organization is LLMEncoder.model -> HF Model, we have to apply PEFT to the inner model
            model.model = initialize_peft(
                model.model,
                lora_r=custom_args.lora_r,
                lora_alpha=2 * custom_args.lora_r,
                lora_dropout=custom_args.lora_dropout,
            )
        else:
            model.model.train()

        tokenizer = model.tokenizer

        data_collator = DefaultCollator(model)

        trainer = LLMEncoderSupervisedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_examples,
            data_collator=data_collator,
            tokenizer=tokenizer,
            loss_function=train_loss,
        )

        if custom_args.stop_after_n_steps is not None:
            trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

        trainer.train()


if __name__ == "__main__":
    main()
