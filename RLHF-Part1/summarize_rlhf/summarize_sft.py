import os

import datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

from summarize_gpt import SummarizePrompt


class SFTSummarize(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        self.split = split
        self.tokenizer = BPETokenizer()
        self.voc = 50257
        self.block_size = block_size
        ds = datasets.load_dataset("CarperAI/openai_summarize_tldr", split=split)
        def drop_long_examples(examples):
            prompts = []
            completions = []
            for prompt, completion in zip(examples['prompt'], examples['label']):
                if self.tokenizer(prompt + completion).size(1) <= block_size + 1:
                    prompts.append(prompt)
                    completions.append(completion)

            return {"prompt": prompts, "completion": completions}

        self.ds = ds.map(drop_long_examples, batched=True, remove_columns=ds.column_names, num_proc=os.cpu_count())


    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        sample = self.ds[idx]
        prompt = self.tokenizer(sample["prompt"]).squeeze(0)
        completion = self.tokenizer(sample["completion"]).squeeze(0)
        toks = torch.cat((prompt, completion))

        # attend to all tokens except the padding tokens
        mask = torch.full((self.block_size + 1,), False, dtype=bool)

        if len(toks) >= self.block_size + 1:
            toks = toks[-self.block_size - 1:]
        else:
            pad = torch.full((self.block_size + 1,), self.tokenizer.eot_token, dtype=torch.long)
            pad[:len(toks)] = toks

            # include a final eot token to predict
            mask[len(toks) + 1:] = True
            toks = pad

        x = toks[:-1]
        y = toks[1:].clone()

        # we only use the completion tokens to learn on
        y[mask[1:]] = -1 # ignore the loss from padding tokens
        # y[:len(prompt)-1] = -1 # and ignore the loss from the prompt tokens
        return x, y, mask[:-1]

def batch_end_callback(trainer):
    model = trainer.model
    model.eval()

    trainer.logger.log("Train", trainer.iter_num, trainer.loss.item())

    if trainer.iter_num % trainer.config.log_every == 0:
        # evaluate both the train and test score
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(valid_loader):
                batch = [x.to(device) for x in batch]
                logits, loss = model(*batch)
                total_loss += loss.item()

        val_loss = total_loss / (i+1)
        trainer.logger.log("Valid", trainer.iter_num, val_loss)
        print(f"E: {trainer.epoch}, iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}, val loss: {val_loss:.5f}")

    if trainer.iter_num % trainer.config.generate_every == 0:
        with torch.no_grad():
            sample_prompt = prompt_ds[0].to(device)
            idx = model.generate(sample_prompt, max_new_tokens=128, do_sample=True, temperature=0.7).cpu()
            for j,generation in enumerate(idx):
                print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    # save the latest model
    if trainer.config.save_every and trainer.iter_num % trainer.config.save_every == 0:
        print("saving model")
        ckpt_path = os.path.join(os.path.curdir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)

    # revert model to training mode
    model.train()


if __name__ == '__main__':
    set_seed(424242)
    torch.set_float32_matmul_precision('high')

    print("===== STARTING PRETRAINING =====")

    # For Logging
    train_idx = []
    train_losses = []
    val_idx = []
    val_losses = []

    valid_iters = 32
    block_size = 1024
    train_ds = SFTSummarize(block_size=block_size, split='train')
    valid_ds = SFTSummarize(block_size=block_size, split='valid')
    prompt_ds = SummarizePrompt(block_size=block_size, split='valid')

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = train_ds.get_vocab_size()
    model_config.block_size = block_size
    model = GPT.from_pretrained("gpt2")

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-6
    train_config.num_workers = 4
    train_config.log_every = 500
    train_config.generate_every = 1000
    train_config.save_every = None
    train_config.epochs = 3
    train_config.batch_size = 8
    train_config.compile = True
    trainer = Trainer(train_config, model, train_ds)

    device = trainer.device
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        num_workers=2,
        batch_size=trainer.config.batch_size * 2,
    )

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    print("===== DONE PRETRAINING =====")

    # Get a validation prompt to test
    sample_prompt = prompt_ds[0].to(device)
    idx = model.generate(sample_prompt, max_new_tokens=128, do_sample=True, top_k=30, stop_at=train_ds.tokenizer.eot_token).cpu()
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))``

    # Plot the losses
    trainer.logger.plot({"Loss": ["Train", "Valid"]}, filename="summarize_sft.png")

    # Save the Model
    torch.save(model.state_dict(), "summarize_sft.pt")
