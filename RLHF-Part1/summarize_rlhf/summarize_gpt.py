# The summarization task differs from the previous happyGPT in a few small ways:
# 1. use the summarization dataset. pretty straight forward containing 100000 summarizations.
#    each datapoint has a prompt and a summary completion. for sft we train only the completion.
#    we also use the gpt-2 tokenizer. also uses mask.
# 2. a proper reward model, not just a heuristic sentiment. the reward model is initialized from
#    pretrained LM and contains a new 'head' layer that predicts the reward. we only train this
#    linear layer of the reward model.
# 3. full ppo optimization to match the instructgpt paper. ppo is very similar to our initial
#    implementation but it prevents the model from updating too far away from the prior distribution
#    using a clipped loss. also uses advantage estimation and a separate value function

import copy
import os

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import trange

from mingpt.bpe import BPETokenizer
from mingpt.logger import Logger
from mingpt.model import GPT
from mingpt.rewards import (RewardModel, ValueModel,
                            calculate_advantage_and_returns)
from mingpt.utils import lr_schedule, masked_mean, set_seed, try_auto_cast


class SummarizePrompt(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        self.split = split
        self.tokenizer = BPETokenizer()
        self.voc = 50257
        self.block_size = block_size
        ds = datasets.load_dataset("CarperAI/openai_summarize_tldr", split=split)
        def drop_long_examples(examples):
            prompts = []
            for prompt in examples['prompt']:
                if self.tokenizer(prompt).size(1) <= block_size:
                    prompts.append(prompt)

            return {"prompt": prompts}

        self.ds = ds.map(
            drop_long_examples,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=os.cpu_count()
        )
        if len(self.ds) > 50000:
            self.ds = ds.select(range(50000))


    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.tokenizer(sample["prompt"])


def validate(valid_ds, model, reward_model, device, max_iters=64):
    reward_model.eval()
    model.eval()

    total_rewards = 0
    count = 0
    total = min(max_iters, len(valid_ds))
    valid_progress_bar = trange(total, desc="Validating", leave=False)
    with torch.no_grad():
        for i in valid_progress_bar:
            prompt = valid_ds[i].to(device)

            completion = model.generate(prompt, max_new_tokens=completion_len, do_sample=True, top_k=30, stop_at=end_of_text)

            reward = reward_model(completion).item()
            total_rewards += reward
            count += 1
            valid_progress_bar.set_postfix(avg_reward=f"{total_rewards/count:.4f}")

            if i < 3:
                print(train_ds.tokenizer.decode(completion[0]), f"\nReward: {reward}\n========\n")

    average_reward = total_rewards / total
    model.train()
    return average_reward


def ppo(model, reward_model, value_model, ref_model, logger):
    # PPO training loop:
    # 1. generate a set of completions, given some prompts for the task
    # 2. calculate the rewards, values and advantages of the completions
    # 3. optimize the models completions based on the rewards using ppo objective
    val_reward = validate(valid_ds, model, reward_model, device)
    print("Initial (SFT) Val reward:", val_reward)

    i = 0
    for epoch in range(n_epochs):
        batch_idxs = torch.randperm(len(train_ds))
        for idx in trange(len(train_ds) // sample_batch_size, desc="iter"):

            # Learning rate schedule
            curr_lr = get_lr(i)
            for pg in optimizer.param_groups:
                pg['lr'] = curr_lr

            # Sample completions given some prompts from the dataset
            # these are the `actions` in the RL sense that the model takes
            with torch.no_grad():
                model.eval()
                value_model.eval()

                original_log_probs = []
                completions = []
                advantages = []
                returns = []
                action_mask = []
                targets = []
                total_reward = 0
                start_idx = idx * sample_batch_size
                for prompt_idx in batch_idxs[start_idx : start_idx + sample_batch_size]:
                    prompt = train_ds[prompt_idx.item()].to(device)

                    # Sample the completions
                    completion = model.generate(prompt, max_new_tokens=completion_len, do_sample=True, top_k=30, stop_at=end_of_text)

                    if completion[0, -1] == end_of_text:
                        # Evaluate and store the rewards for the last token
                        reward = reward_model(completion).unsqueeze(-1)

                    else:
                        # If there is no eot token, hardcode a negative reward
                        reward = torch.tensor([[-1.0]], device=device)

                    total_reward += reward.item()

                    completion_minus_1, target = completion[:, :-1], completion[:, 1:]

                    # Store the model's original log prob (could be merged into the generate fn)
                    original_log_prob = model.log_probs(completion_minus_1, target)

                    # Reference logprobs
                    ref_log_prob = ref_model.log_probs(completion_minus_1, target)

                    # Calculate values, returns and advantages
                    values = value_model(completion)

                    # Calculate the advantage for our policy gradient
                    # Include the kl score to reduce overfitting
                    # the kl reward here could be kept up to date with the policy network
                    # inside the ppo updates below for a better regularization effect
                    kl = original_log_prob - ref_log_prob
                    score = torch.cat((- kl_beta * kl, reward), dim=1)
                    advantage, single_return = calculate_advantage_and_returns(score, values, gamma=gamma, lambd=lambd)

                    # Pad the values up to block_size with zeros
                    pad = torch.zeros(1, block_size - advantage.size(1), device=advantage.device, dtype=advantage.dtype)
                    advantages.append(torch.cat((advantage, pad), dim=1))
                    returns.append(torch.cat((single_return, pad), dim=1))

                    # pad the log probs with 1 extra 0
                    pad_plus_1 = torch.zeros(1, block_size - original_log_prob.size(1), device=advantage.device, dtype=advantage.dtype)
                    original_log_probs.append(torch.cat((original_log_prob, pad_plus_1), dim=1))

                    # Pad the tokens with longs
                    pad = torch.zeros(1, block_size - completion.size(1), device=completion.device, dtype=completion.dtype)
                    completions.append(torch.cat((completion, pad), dim=1))
                    pad = torch.zeros(1, block_size - target.size(1), device=target.device, dtype=target.dtype)
                    targets.append(torch.cat((target, pad), dim=1))

                    # The action mask is only the generated part of the completion
                    mask = torch.zeros(1, block_size, device=advantage.device, dtype=advantage.dtype)
                    mask[:, prompt.size(1):completion.size(1)] = 1
                    action_mask.append(mask)

            # Stack the values into a batch
            advantages = torch.cat(advantages)
            returns = torch.cat(returns)
            completions = torch.cat(completions)
            original_log_probs = torch.cat(original_log_probs)
            action_mask = torch.cat(action_mask)
            targets = torch.cat(targets)


            # Do the PPO update on the batch of data several times
            model.train()
            value_model.train()
            for _ in range(n_updates):
                b_inds = torch.randperm(sample_batch_size)
                for start in range(0, sample_batch_size, train_batch_size):
                    end = start + train_batch_size

                    # Grab the mini-batches
                    mb_inds = b_inds[start:end]
                    mb_completion = completions[mb_inds]
                    mb_target = targets[mb_inds]
                    mb_original_logps = original_log_probs[mb_inds]
                    mb_advantages = advantages[mb_inds]
                    mb_returns = returns[mb_inds]
                    mb_action_mask = action_mask[mb_inds]

                    # TODO: Masked normalize advantages
                    # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    with try_auto_cast(device):
                        # Forward pass through the latest model
                        log_probs = model.log_probs(mb_completion, mb_target)

                        # Policy loss
                        logratio = log_probs - mb_original_logps
                        ppo_ratio = logratio.exp()
                        pg_loss1 = -mb_advantages * ppo_ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ppo_ratio, 0.8, 1.2)
                        pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_action_mask, dim=1).mean()

                        # Value loss
                        new_value = value_model(mb_completion)
                        v_loss = 0.5 * masked_mean((new_value - mb_returns) ** 2, mb_action_mask, dim=1).mean()

                        loss = pg_loss + 0.1 * v_loss
                        loss = loss / grad_accum_steps

                    loss.backward()

                policy_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                value_grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), grad_norm_clip)
                optimizer.step()
                model.zero_grad()
                value_model.zero_grad()


            clipfrac = ((ppo_ratio - 1.0).abs() > 0.2).float().mean().item()
            avg_reward = total_reward / sample_batch_size
            logger.log("Clip Frac", i, clipfrac)
            logger.log("Reward", i, avg_reward)
            logger.log("Value Loss", i, v_loss.item())
            logger.log("KL", i, kl.mean().item())
            logger.log("Policy Grad Norm", i, policy_grad_norm.item())
            logger.log("Value Grad Norm", i, value_grad_norm.item())

            if i % 20 == 0:
                val_reward = validate(valid_ds, model, reward_model, device)
                logger.log("Val Reward", i, val_reward)
                print(f"Iter: {i}, Avg reward: {avg_reward:.3f}, KL: {kl.mean().item():.3f}, Value Loss: {v_loss.item():.4f}, Grad Norm: {policy_grad_norm:.2f}, Vf grad norm: {value_grad_norm:.2f}, Val reward: {val_reward:.3f}")

                # intermediate plots help with debugging!
                logger.plot({"Reward": ["Reward", "Val Reward"], "Value Loss": ["Value Loss"]}, filename="summarize_rl_rewards.png")
                logger.plot({"Gradient Norm": ["Policy Grad Norm", "Value Grad Norm"], "KL Div": ["KL"]}, filename="summarize_rl_metrics.png")

                torch.save(uncompiled_model.state_dict(), "summarize_rl.pt")

            i += 1


if __name__ == "__main__":
    set_seed(424242)
    torch.set_float32_matmul_precision('high')

    # ------
    # Config
    # ------

    # Transformer context len
    # For GPT-2, should be 1024
    block_size = 1024

    # completion + prompt <= block_size
    completion_len = 80
    max_prompt_len = 512

    model = GPT.get_default_config()
    model.model_type = "gpt2"
    model.n_layer = 12
    model.n_head = 12
    model.n_embd = 768
    model.vocab_size = 50257
    model.model_type = None
    model.block_size = block_size

    reward_model = RewardModel(model)
    value_model = ValueModel(model)
    model = GPT(model)

    # Load reward, value, and model from weights!
    model.load_state_dict(torch.load("summarize_sft.pt", map_location='cpu'))
    reward_model.load_state_dict(torch.load("reward_model.pt", map_location='cpu'))
    value_model.load_state_dict(torch.load("reward_model.pt", map_location='cpu'))

    # reference model is the finetuned SFT model
    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)
    ref_model.eval()

    reward_model.requires_grad_(False)
    reward_model.eval()

    # PPO hyperparams
    sample_batch_size = 64 # Number of completions to sample
    train_batch_size = 8 # OAI uses equal training and sampling batches of 64 (we'll use whatever fits on the GPU!)
    grad_accum_steps = sample_batch_size // train_batch_size
    max_learning_rate = 3e-6
    grad_norm_clip = 1.0
    kl_beta = 0.02
    n_updates = 2
    gamma = 1
    lambd = 0.95
    n_epochs = 1

    # Logging
    logger = Logger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    value_model.to(device)
    reward_model.to(device)
    ref_model.to(device)
    print("Running on device", device)

    uncompiled_model = model
    uncompiled_value_model = value_model

    # compile the model
    model = torch.compile(model)
    reward_model = torch.compile(reward_model)
    value_model = torch.compile(value_model)
    ref_model = torch.compile(ref_model)

    # Prompt datasets
    train_ds = SummarizePrompt('train', block_size=max_prompt_len)
    valid_ds = SummarizePrompt('valid', block_size=max_prompt_len)
    end_of_text = train_ds.tokenizer.eot_token
    print("train ds:", len(train_ds), "val ds:", len(valid_ds))

    # Can set separate lrs for policy and value fn
    total_iters = len(train_ds) // sample_batch_size * n_epochs
    get_lr = lr_schedule(max_learning_rate, max_iters=total_iters)
    optim_groups = [{'params': model.parameters()}, {'params': value_model.parameters()}]
    optimizer = torch.optim.AdamW(optim_groups, lr=get_lr(0), betas=(0.9, 0.95), fused=torch.cuda.is_available(), weight_decay=0.0)

    # --------
    # Run PPO!
    # --------
    # I'm just using the global config vars here :)
    ppo(model, reward_model, value_model, ref_model, logger)

    torch.save(uncompiled_model.state_dict(), "summarize_rl.pt")
    torch.save(uncompiled_value_model.state_dict(), "value_model.pt")

    # Plot the results
    logger.plot({"Reward": ["Reward", "Val Reward"], "Value Loss": ["Value Loss"]}, filename="summarize_rl_rewards.png")
    logger.plot({"Gradient Norm": ["Policy Grad Norm", "Value Grad Norm"], "KL Div": ["KL"], "Clip Fraction": ["Clip Frac"]}, filename="summarize_rl_metrics.png")
