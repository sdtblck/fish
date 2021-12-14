from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
from argparse import ArgumentParser
# wandb agent eleutherai/uncategorized/rem68dr3
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from fish import FisherMask, JsonlDataset, enable_grad_checkpointing, AnnealingLR
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging, sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument(
        "--dataset_url_or_path",
        type=str,
        default="http://eaidata.bmk.sh/data/enron_emails.jsonl.zst",
    )
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--disable_fisher_mask", action="store_true")
    parser.add_argument("--keep_ratio", type=float, default=0.005)
    parser.add_argument("--fisher_n_samples", type=int, default=1024)
    parser.add_argument("--compute_fisher_on_cpu", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--max_train_steps", type=int, default=None)
    return parser.parse_args()


def get_model(model_name):
    model_name = model_name.lower()
    if model_name in ["125m", "EleutherAI/gpt-neo-125M".lower()]:
        return GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    elif model_name in ["2.7b", "EleutherAI/gpt-neo-2.7B".lower()]:
        return GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    elif model_name in ["1.3b", "EleutherAI/gpt-neo-1.3B".lower()]:
        return GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    else:
        raise ValueError(f"Model {model_name} not found")


if __name__ == "__main__":

    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = JsonlDataset(
        args.dataset_url_or_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    train_dataset, valid_dataset = random_split(
        dataset,
        [
            int(len(dataset) * args.train_val_split),
            len(dataset) - int(len(dataset) * args.train_val_split),
        ],
    )

    model = get_model(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not args.disable_fisher_mask:
        fisher_mask = FisherMask(
            model, device="cpu" if args.compute_fisher_on_cpu else device
        )  # compute fisher info on cpu if you have limited gpu memory

        # masks are returned as a dict of module name to indices to mask, also stored in fisher_mask.masks
        inds_to_mask = fisher_mask.calculate_mask(
            train_dataset,
            keep_ratio=args.keep_ratio,
            n_samples=args.fisher_n_samples,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
        )

        # apply the masks (done with torch.nn.utils.parametrize), original params are frozen
        fisher_mask.apply_masks()

    # do fine-tuning
    model.train()
    enable_grad_checkpointing(model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    args.num_params = sum(p.numel() for p in trainable_parameters)
    logger.info(f"Finetuning with {args.num_params} params")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    valid_dataloader = iter(
        DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
        )
    )
    total_iters = len(dataloader) * args.epochs
    lr_scheduler = AnnealingLR(
        optimizer,
        args.lr,
        warmup_iter=int(args.warmup * total_iters),
        total_iters=total_iters,
    )

    if WANDB_AVAILABLE:
        wandb.init(project="fisher_finetune", config=args)
        log_fn = lambda *args, **kwargs: wandb.log(*args, **kwargs)
    else:
        log_fn = lambda *args, **kwargs: None

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Train epoch {epoch} - loss: ")

        for i, batch in enumerate(pbar):

            model.zero_grad()
            batch = batch.to(device)
            labels = batch.clone().to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(batch, labels=labels)
            loss = outputs.loss
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(f"Train epoch {epoch} - loss: {loss.item():.4f}")
            log_fn({"train/loss": loss.item()}, step=epoch * len(dataloader) + i)
            log_fn(
                {"train/lr": optimizer.param_groups[0]["lr"]},
                step=epoch * len(dataloader) + i,
            )
            lr_scheduler.step()

            if i % args.eval_every == 0 or (
                epoch == args.epochs - 1 and i == len(dataloader) - 1
            ):

                # evaluate on validation set
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    pbar = tqdm(
                        range(args.eval_steps),
                        desc=f"Validation epoch {epoch} - loss: ",
                    )
                    for _ in pbar:
                        batch = next(valid_dataloader).to(device)
                        labels = batch.clone().to(device)
                        labels[labels == tokenizer.pad_token_id] = -100

                        outputs = model(batch, labels=labels)
                        loss = outputs.loss
                        valid_loss += loss

                        pbar.set_description(
                            f"Validation epoch {epoch} - loss: {loss.item():.4f}"
                        )

                    valid_loss /= args.eval_steps
                    logger.info(f"Validation loss: {valid_loss}")
                    log_fn({"eval/loss": loss.item()}, step=epoch * len(dataloader) + i)

                    # generate some samples
                    if WANDB_AVAILABLE:
                        logger.info("Generating samples...")
                        start = tokenizer(
                            "<|endoftext|>", return_tensors="pt"
                        ).input_ids.to(device)
                        sample_outputs = model.generate(
                            start,
                            do_sample=True,
                            max_length=64,
                            top_p=0.9,
                            temperature=0.5,
                            num_return_sequences=args.batch_size,
                        )
                        sample_outputs = [
                            [tokenizer.decode(ids)] for ids in sample_outputs
                        ]
                        log_fn(
                            {
                                "samples": wandb.Table(
                                    data=sample_outputs, columns=["Text"]
                                )
                            },
                            step=epoch * len(dataloader) + i,
                        )

                model.train()

            if args.max_train_steps is not None and i >= args.max_train_steps:
                break
