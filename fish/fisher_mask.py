import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
from tqdm import tqdm
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
import logging
import sys
from .utils import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__file__)


def empirical_fisher(
    model,
    data_iterator,
    n_samples=100,
    labels_iterator=None,
    device=None,
    pad_token_id=None,
    loss_fn=cross_entropy_loss,
    **model_kwargs,
):
    was_training = model.training
    orig_device = next(model.parameters()).device
    if device is not None:
        model.to(device)
    was_grad_checkpointing = grad_checkpointing_is_enabled(model)
    enable_grad_checkpointing(model)

    # dict to store fisher information values
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param).to(device)

    samples_seen = 0
    pbar = tqdm(None, desc="Computing empirical fisher information...", total=n_samples)
    for i, inputs in enumerate(data_iterator):
        if labels_iterator is not None:
            labels = next(labels_iterator)
        else:
            # assume this is an auto-regressive model
            # labels should be inputs shifted by one
            assert pad_token_id is not None
            labels = inputs.clone()
            labels[labels == pad_token_id] = -100

        if device is not None:
            inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs, **model_kwargs)

        logits = getattr(outputs, "logits")
        if logits is None:
            logits = outputs

        loss = getattr(outputs, "loss")
        if loss is None:
            loss = loss_fn(logits, labels)

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] += torch.square(param.grad).data

        model.zero_grad()

        bs = inputs.shape[0]
        samples_seen += bs
        pbar.update(bs)
        if samples_seen >= n_samples:
            break
        if samples_seen >= n_samples:
            break

    if not was_grad_checkpointing:
        disable_grad_checkpointing(model)
    model.train(was_training)
    model.to(orig_device)
    return OrderedDict(
        sorted(fim.items(), key=lambda item: item[1].mean(), reverse=True)
    )


class Mask(nn.Module):
    def __init__(self, orig_module, inds, attr_to_parametrize="weight"):
        super().__init__()
        orig_param = getattr(orig_module, attr_to_parametrize)
        self.orig_shape = orig_param.shape
        assert inds.max() < orig_param.numel()

        self.register_buffer("inds", inds)

        self.vals = torch.nn.Parameter(orig_param.flatten()[inds])

        # freeze original params + register parameterization
        orig_param.requires_grad = False
        parametrize.register_parametrization(orig_module, attr_to_parametrize, self)
        self.to(orig_param.device)

    def forward(self, X):
        assert X.shape == self.orig_shape, "Input shape must match original shape"
        # put inds back into X
        X = torch.scatter(X.flatten(), 0, self.inds, self.vals)
        return X.view(self.orig_shape)


class FisherMask:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.masks = None

    def calculate_mask(
        self,
        train_dataset,
        keep_ratio,
        n_samples=1000,
        pad_token_id=None,
        batch_size=None,
        approx_type="empirical",
    ):

        data_loader = DataLoader(
            train_dataset, batch_size=batch_size or 1, shuffle=False, num_workers=1
        )

        if approx_type == "empirical":
            fisher_method = empirical_fisher
        else:
            raise NotImplementedError

        fisher_info = fisher_method(
            self.model,
            data_loader,
            n_samples=n_samples,
            labels_iterator=None,
            device=self.device,
            pad_token_id=pad_token_id,
        )

        # get total number of parameters from returned gradients
        n_params = sum(g.numel() for g in fisher_info.values())

        keep_n = int(keep_ratio * n_params)
        logger.info(f"Expected Masked # params: {keep_n}")

        tensors = torch.cat([g.flatten() for g in fisher_info.values()], dim=0)

        values, _ = torch.topk(tensors, k=keep_n)

        # get min value - we can use this to make the mask for each module
        min_val = values.min()

        inds_to_mask = {}

        for name, grad in fisher_info.items():
            # get the indices to mask
            inds = (grad.flatten() >= min_val).nonzero().flatten()
            inds_to_mask[name] = inds

        # delete redundant data & clear cuda cache
        del values, tensors
        del fisher_info
        torch.cuda.empty_cache()

        # return masks
        self.masks = inds_to_mask
        return inds_to_mask

    def apply_masks(self):

        # go through each param and mask it
        for name, inds in self.masks.items():

            if inds.numel() == 0:
                continue

            # we assume that the parent module of each param is the penultimate module
            # attr in the name, and the last attr is the weight/bias

            module_attrs = name.split(".")[:-1]
            param_attr = name.split(".")[-1]
            orig_module = getattr_recursive(self.model, module_attrs)
            Mask(orig_module, inds, param_attr)

        logger.info(f"Masking complete!")
        logger.info(
            f"Original # params: {sum(p.numel() for p in self.model.parameters())}"
        )
        logger.info(
            f"Masked # params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
