## Summarize RLHF

Finetune GPT2 124M to generate summaries via SFT and RLHF.

See [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593) for more.

```bash
# Run supervised finetuning
python3 summarize_sft.py

# Train reward model
python3 summarize_reward_model.py

# Run RLHF using SFT checkpoint and reward model
python3 summarize_gpt.py
```
