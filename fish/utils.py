import os
import torch.nn.functional as F
import torch
import logging
import sys
import json
from torch.utils.data import Dataset
import math
from pathlib import Path

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__file__)


def cross_entropy_loss(logits, labels):
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)  # calculate loss in fp32

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )

    logits = logits.to(orig_dtype)
    return loss


def _set_grad_checkpointing(model, value, use_cache=None):
    config = getattr(model, "config")
    if config is None:
        return
    setattr(config, "gradient_checkpointing", value)
    if use_cache is not None:
        setattr(config, "use_cache", use_cache)
    else:
        setattr(config, "use_cache", not value)
    model.train(value)


def grad_checkpointing_is_enabled(model):
    config = getattr(model, "config", None)
    if config is None:
        return False
    return getattr(config, "gradient_checkpointing", False)


def enable_grad_checkpointing(model):
    # for huggingface models
    # for whatever reason, gradient checkpointing is off by default, and even small models
    # use a ridiculous amount of memory.
    _set_grad_checkpointing(model, True)


def disable_grad_checkpointing(model):
    _set_grad_checkpointing(model, False)


def getattr_recursive(obj, attrs):
    if isinstance(attrs, str):
        attrs = attrs.split(".")
    else:
        assert isinstance(attrs, (list, tuple))
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


class JsonlDataset(Dataset):
    def __init__(
        self,
        path_or_url: str,
        tokenizer,
        max_length=None,
        field="text",
        pad_token_id=None,
        cache_dir=".datasets",
    ):
        self.field = field
        self.data = []
        if path_or_url.startswith("http"):
            import urllib.request

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True, parents=True)

            # download the file to cache
            dataset_path = cache_dir / Path(path_or_url).name
            is_zst = dataset_path.suffix == ".zst"
            if is_zst:
                decompressed_path = dataset_path.with_suffix("")
            if not dataset_path.exists() or (is_zst and not decompressed_path.exists()):
                os.system(f"wget -O {dataset_path} {path_or_url}")

        else:
            dataset_path = Path(path_or_url)
            is_zst = dataset_path.suffix == ".zst"
            if is_zst:
                decompressed_path = dataset_path.with_suffix("")

        # decompress the file if needed
        if is_zst:
            if not decompressed_path.exists():
                os.system(f"zstd -d {dataset_path} -o {decompressed_path}")
            dataset_path = decompressed_path

        assert dataset_path.exists() and dataset_path.suffix == ".jsonl"
        with open(dataset_path) as f:
            for i in f.readlines():
                self.data.append(json.loads(i))

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            if pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError(
                    "pad_token_id must be specified if tokenizer.pad_token_id is None"
                )

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        txt = d[self.field]
        return self.tokenizer.encode(
            txt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).squeeze(0)


class AnnealingLR:
    """Anneals & warms up the learning rate."""

    def __init__(
        self,
        optimizer,
        lr,
        warmup_iter,
        total_iters,
        current_iter=None,
        decay_style="cosine",
        min_lr=None,
        use_checkpoint_lr_scheduler=True,
        override_lr_scheduler=False,
    ):

        # Class values.
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr if min_lr is not None else lr * 0.1
        self.warmup_iter = warmup_iter
        self.current_iter = current_iter if current_iter is not None else 0
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.decay_style = decay_style
        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, (
                "both override and " "use-checkpoint are set."
            )
        # Set the learning rate
        self.step(self.current_iter)

        print("> learning rate decay style: {}".format(self.decay_style))

    def get_lr(self):
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        num_iters_ = min(self.current_iter, self.end_iter - self.warmup_iter)
        # Warmup.
        if self.warmup_iter > 0 and self.current_iter <= self.warmup_iter:
            return float(self.lr) * num_iters_ / self.warmup_iter

        num_iters_ = num_iters_ - self.warmup_iter
        if self.decay_style == "linear":
            lr = self.lr * (self.end_iter - num_iters_) / self.end_iter
        elif self.decay_style == "cosine":
            lr = self.lr / 2.0 * (math.cos(math.pi * num_iters_ / self.end_iter) + 1)
        elif self.decay_style == "exponential":
            # exp(-0.693) = 1/2
            lr = self.lr * math.exp(-0.693 * num_iters_ / self.end_iter)
        else:
            lr = self.lr
        return max(lr, self.min_lr)

    def step(self, step_num=None):
        """Set lr for all parameters groups."""
        if step_num is None:
            step_num = self.current_iter + 1
        self.current_iter = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def state_dict(self):
        state_dict = {
            "lr": self.lr,
            "warmup_iter": self.warmup_iter,
            "num_iters": self.current_iter,
            "decay_style": self.decay_style,
            "end_iter": self.end_iter,
            "min_lr": self.min_lr,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            print(" > overriding {} value to {}".format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, (
                "AnnealingLR: class input value"
                "and checkpoint values for {} do not match".format(name)
            )
        print(" > using checkpoint value {} for {}".format(sd_value, name))
        return sd_value

    def load_state_dict(self, sd):

        self.lr = self._check_and_set(self.lr, sd["lr"], "learning rate")
        self.min_lr = self._check_and_set(
            self.min_lr, sd["min_lr"], "minimum learning rate"
        )
        self.warmup_iter = self._check_and_set(
            self.warmup_iter, sd["warmup_iter"], "warmup iterations"
        )
        self.end_iter = self._check_and_set(
            self.end_iter, sd["end_iter"], "total number of iterations"
        )
        self.decay_style = self._check_and_set(
            self.decay_style, sd["decay_style"], "decay style"
        )

        self.current_iter = sd["num_iters"]
        self.step(self.current_iter)
