import copy
import os

import datasets
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.char_tokenizer import CharTokenizer
from mingpt.logger import Logger
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


class TweetDataset(Dataset):
    # Character level tweet dataset: https://huggingface.co/datasets/mteb/tweet_sentiment_extraction
    def __init__(self, block_size, split="train", label=None, tokenizer=None):
        assert split in ["train", "test"]
        self.block_size = block_size

        eot = "⏎"
        def chunk_examples(examples):
            chunks = [(eot+text+eot) for text, lbl in zip(examples['text'], examples['label_text']) if len(text) > 0 and (label is None or lbl == label)]
            return {"content": chunks}

        dataset = datasets.load_dataset("mteb/tweet_sentiment_extraction", split=split)
        self.dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)

        if tokenizer is None:
            token_text = "".join(row["content"] for row in self.dataset)
            self.tokenizer = CharTokenizer(token_text)
        else:
            self.tokenizer = tokenizer

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chunk = self.dataset[idx]["content"]
        toks = self.tokenizer(chunk)
        assert len(toks) > 0
        if len(toks) >= self.block_size + 1:
            toks = toks[:self.block_size + 1]
        else:
            pad = torch.full((self.block_size + 1,), self.tokenizer.pad_token, dtype=torch.long)
            pad[:len(toks)] = toks
            toks = pad

        x = toks[:-1]
        y = toks[1:].clone()
        y[y == self.tokenizer.pad_token] = -1

        # Our mask is true for padding/unused tokens
        # to match the causal masking inside the minGPT model
        attn_mask = x == self.tokenizer.pad_token
        return x, y, attn_mask

class SentimentRewardModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sid = SentimentIntensityAnalyzer()

    def sentiment(self, sentence: str) -> float:
        return self.sid.polarity_scores(sentence)['compound']

    def __call__(self, tokens):
        bs = tokens.shape[0]
        rewards = torch.zeros(bs)
        for i in range(bs):
            sentence = self.tokenizer.decode(tokens[i])
            rewards[i] = self.sentiment(sentence)

        return rewards.to(tokens.device)


def batch_end_callback(trainer):
    model = trainer.model
    model.eval()

    trainer.logger.log("Train", trainer.iter_num, trainer.loss.item())

    if trainer.iter_num % trainer.config.log_every == 0:
        # evaluate both the train and test score
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(valid_loader):
                x, y, mask = [x.to(trainer.device) for x in batch]
                logits, loss = model(x, y, attention_mask=mask)
                total_loss += loss.item()

        val_loss = total_loss / (i+1)
        trainer.logger.log("Valid", trainer.iter_num, val_loss)
        print(f"E: {trainer.epoch}, iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}, val loss: {val_loss:.5f}")

    if trainer.iter_num % trainer.config.generate_every == 0:
        with torch.no_grad():
            context = train_ds.tokenizer(TEST_PROMPT)[None]
            x = context.to(trainer.device)
            y = model.generate(x, 140, temperature=1.0, do_sample=True, top_k=10)[0]
            completion = train_ds.tokenizer.decode(y)
            print(completion)

    # revert model to training mode
    model.train()


if __name__ == '__main__':
    set_seed(424242)

    print("===== STARTING PRETRAINING =====")

    TEST_PROMPT = "⏎I think "
    valid_iters = 32
    block_size = 32
    train_ds = TweetDataset(block_size, split='train')
    valid_ds = TweetDataset(block_size, split='test', tokenizer=train_ds.tokenizer)
    end_of_text = train_ds.tokenizer.eot_token

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = train_ds.get_vocab_size()
    model_config.block_size = block_size
    model = GPT(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-3
    train_config.num_workers = 0
    train_config.log_every = 500
    train_config.generate_every = 1000
    train_config.epochs = 10
    train_config.compile = True
    trainer = Trainer(train_config, model, train_ds)
    device = trainer.device

    sample_size = 3
    sample_prompt = torch.full((sample_size, 1), end_of_text, dtype=torch.long, device=device)

    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        batch_size=trainer.config.batch_size * 2,
    )

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    print("\n===== DONE PRETRAINING =====")

    idx = model.generate(sample_prompt, max_new_tokens=80, top_k=30, do_sample=True)
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    # Plot the losses
    trainer.logger.plot({"Loss": ["Train", "Valid"]}, filename='happy_gpt_pretrain.png')

    print("\n===== STARTING SFT =====")

    sft_dataset = TweetDataset(block_size, split='train', label='positive', tokenizer=train_ds.tokenizer)
    sft_valid = TweetDataset(block_size, split='test', label='positive', tokenizer=train_ds.tokenizer)

    valid_loader = DataLoader(
        sft_valid,
        shuffle=False,
        batch_size=trainer.config.batch_size * 2,
    )

    train_config.learning_rate = 3e-4
    train_config.epochs = 6

    print(len(sft_dataset), "SFT tweets")
    trainer = Trainer(train_config, model, sft_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    print("\n===== DONE SFT =====")
    idx = model.generate(sample_prompt, max_new_tokens=80, top_k=30, do_sample=True)
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    trainer.logger.plot({"Loss": ["Train", "Valid"]}, filename='happy_gpt_sft.png')

    print("\n===== STARTING RL =====")

    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)
    ref_model.eval()
    reward_model = SentimentRewardModel(train_ds.tokenizer)
    batch_size = 32
    num_iters = 100
    optim_groups = model.parameters()
    learning_rate = 3e-4
    grad_norm_clip = 1.0
    kl_beta = 0.1
    model.to(device)
    logger = Logger()

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    prompt = torch.full((batch_size, 1), end_of_text, dtype=torch.long, device=device)

    for i in range(num_iters):
        # ------------------------------------------------------------
        # Sample Data
        # These are the `actions` in the RL sense that the model takes
        # ------------------------------------------------------------
        with torch.no_grad():
            model.eval()
            # Sample new tweets
            completion = model.generate(prompt, max_new_tokens=block_size, temperature=1.0, do_sample=True, top_k=30)
            model.train()

            # Evaluate the rewards
            rewards = reward_model(completion)

            # Reference logprobs
            ref_log_probs = ref_model.log_probs(completion[:, :-1], completion[:, 1:])

        log_probs = model.log_probs(completion[:, :-1], completion[:, 1:])
        kl = log_probs - ref_log_probs
        score = - kl_beta * kl

        # NOTE: OAI seems to only apply this at the last token, because they calculate the advantage later on.
        # We instead use the trajectory return with gamma=1 as the weight in the policy gradient.
        score += rewards[:, None]
        score = score.detach()

        # ---------------------------------
        # Train via Vanilla Policy Gradient
        # ---------------------------------

        # Put the reward into the kl div and multiply by the logprobs
        pg_loss = -torch.mean(score * log_probs)
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        # -------
        # Logging
        # -------

        logger.log("reward", i, rewards.mean().item())
        logger.log("kl", i, kl.mean().item())

        if i % 10 == 0:
            print(f"Iter: {i}, Avg reward: {rewards.mean().item():.4f}, KL: {kl.mean().item():.4f}")

        if i % 20 == 0:
            model.eval()
            idx = model.generate(sample_prompt, max_new_tokens=80, do_sample=True, top_k=30).cpu()
            for j,generation in enumerate(idx):
                print(f"Generation {j}:", train_ds.tokenizer.decode(generation))
            model.train()

    print("\n===== DONE RL =====")
    idx = model.generate(sample_prompt, max_new_tokens=128, top_k=10, do_sample=True)
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    print("\n===== SAVING MODEL =====")
    ckpt_path = os.path.join(os.path.curdir, "happy_gpt_pg.pt")
    torch.save(model.state_dict(), ckpt_path)

    logger.plot({"Reward": ["reward"], "KL Divergence": ["kl"]}, filename='happy_gpt_rl_pg.png')
