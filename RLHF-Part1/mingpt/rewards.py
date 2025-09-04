import torch
from torch import nn

from mingpt.model import Transformer


def calculate_advantage_and_returns(rewards, values, gamma, lambd):
    """Calculate the GAE estimate of the advantage."""
    lastgaelam = 0
    advantages = torch.zeros_like(rewards)
    gen_len = rewards.size(1)

    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        advantages[:, t] = lastgaelam = delta + gamma * lambd * lastgaelam
    returns = advantages + values

    return advantages, returns


def shift_completions_and_mask(prompt, completion, end_of_text, block_size):
    """
    This method processes the prompts and completions by stacking and padding them on the right.

    Steps:
    1. Extract the first non-padding token from the left-padded prompts.
    2. Identify the last token in the completions, truncating at the first end-of-text token.
    3. Concatenate the prompts and completions, right-padding the results with end-of-text tokens.
    4. Create attention masks and action masks for the concatenated sequences.

    Args:
        prompt (torch.Tensor): Tensor containing the prompt tokens, left-padded.
        completion (torch.Tensor): Tensor containing the completion tokens.
        end_of_text (int): Token ID representing the end-of-text.

    Returns:
        completions (torch.Tensor): Concatenated and padded prompt and completion sequences.
        attn_mask (torch.Tensor): Attention mask indicating which tokens should be attended to.
        action_mask (torch.Tensor): Action mask indicating which tokens are part of the completion.
    """
    # Get the first prompt token since they're left padded
    prompt_padding = (prompt != end_of_text).long().argmax(dim=1)

    # Get the last completion token
    token_ends = (completion == end_of_text).long().argmax(dim=1)

    # If there isn't a last completion token, select the end
    token_ends[torch.logical_and(token_ends == 0, completion[:, 0] != end_of_text)] = completion.size(1)

    prompt_size = prompt.size(1)

    # cat the prompt and completions and right pad the results
    # creating masks for the attn and the completion parts separately
    completions = []
    action_masks = []
    attn_masks = []
    for i, (prompt_start, completion_end) in enumerate(zip(prompt_padding, token_ends)):
        x = prompt[i, prompt_start:]
        c = completion[i,:completion_end]

        # we pad eot tokens to block_size + 1
        pad_size = block_size - prompt_size + prompt_start.item() - completion_end.item() + 1
        padding = torch.full((pad_size,), end_of_text, dtype=prompt.dtype, device=prompt.device)
        comp = torch.cat((x, c, padding))
        completions.append(comp)

        # true for values that are ignored in attention, opposite for the completion
        attn_mask = torch.full((block_size,), False, dtype=bool, device=prompt.device)
        if pad_size > 2:
            attn_mask[-pad_size+2:] = True # include the first eot token
        attn_masks.append(attn_mask)

        action_mask = attn_mask.clone()
        action_mask[:x.size(0)] = True
        action_mask = torch.logical_not(action_mask)
        action_masks.append(action_mask)

    action_mask = torch.stack(action_masks)
    attn_mask = torch.stack(attn_masks)
    completions = torch.stack(completions)
    target = completions[:, 1:].clone()

    return completions[:, :-1], attn_mask, action_mask, target


class ValueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config)
        self.prediction_head = nn.Linear(config.n_embd, 1, bias=True)
        torch.nn.init.normal_(self.prediction_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction_head.bias)

    def forward(self, toks, attn_mask=None):
        x = self.transformer(toks, attention_mask=attn_mask) # (b, t, n_embd)
        rewards = self.prediction_head(x).squeeze(-1) # (b,)

        return rewards

# The reward model looks more complicated, but we are just batching positive and
# negative responses together and calculating the loss.
# We also gather the reward prediction from the last non-masked token.
class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config)
        self.prediction_head = nn.Linear(config.n_embd, 1)
        torch.nn.init.normal_(self.prediction_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction_head.bias)

    def forward(self, toks, attn_mask=None, positive_tokens=None, positive_mask=None):
        if positive_tokens is not None:
            assert toks.size(0) == positive_tokens.size(0)
            toks = torch.cat((toks, positive_tokens))

        if positive_mask is not None:
            assert attn_mask is not None
            attn_mask = torch.cat((attn_mask, positive_mask))

        x = self.transformer(toks, attention_mask=attn_mask) # (b, t, n_embd)

        if attn_mask is None:
            reward_idxs = -1
        else:
            # Gets the last non-masked value
            reward_idxs = attn_mask.size(1) - torch.flip(attn_mask, dims=[1]).to(torch.int64).argmin(dim=1) - 1

        x = x[torch.arange(x.size(0)), reward_idxs] # (b, n_embd)
        rewards = self.prediction_head(x).squeeze(1) # (b,)

        if positive_tokens is not None:
            s = positive_tokens.size(0)
            rejected = rewards[:s]
            chosen = rewards[s:]
            loss = -torch.mean(nn.functional.logsigmoid(chosen - rejected))
            acc = (chosen > rejected).float().detach().mean()
            return loss, acc

        return rewards
