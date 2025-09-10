import torch
import gradio as gr
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_WEIGHTS_PATH = "summarize_rl.pt"
BLOCK_SIZE = 1024
COMPLETION_LEN = 80 # Max number of tokens to generate for the summary

print(f"Using device: {DEVICE}")

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")

# 1. Initialize Tokenizer
tokenizer = BPETokenizer()
end_of_text_token_id = tokenizer.eot_token

# 2. Configure the GPT model (must match the training configuration)
model_config = GPT.get_default_config()
model_config.model_type = "gpt2"
model_config.n_layer = 12
model_config.n_head = 12
model_config.n_embd = 768
model_config.vocab_size = 50257
model_config.block_size = BLOCK_SIZE
model_config.model_type = None # This was set to None in your script

# 3. Load the fine-tuned model
model = GPT(model_config)
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    # Exit if model can't be loaded
    exit()

model.to(DEVICE)
model.eval() # Set the model to evaluation mode

# --- Prediction Function ---
def generate_summary(prompt_text):
    """
    This function takes a text prompt, tokenizes it, generates a summary
    using the model, and decodes the result back to text.
    """
    if not prompt_text:
        return "Please enter some text to summarize."

    print(f"Received prompt: {prompt_text[:50]}...")

    # Tokenize the input prompt
    prompt_tokens = tokenizer(prompt_text).to(DEVICE)

    # Generate the completion (summary)
    with torch.no_grad():
        completion_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=COMPLETION_LEN,
            do_sample=True,
            top_k=30,
            stop_at=end_of_text_token_id
        )

    # Decode the generated tokens into text
    # The output includes the prompt, so we extract just the generated part.
    prompt_len = prompt_tokens.size(1)
    generated_part = completion_tokens[0, prompt_len:]
    summary_text = tokenizer.decode(generated_part)

    print(f"Generated summary: {summary_text}")
    return summary_text

# --- Launch the Gradio Web App ---
print("Launching Gradio interface...")

iface = gr.Interface(
    fn=generate_summary,
    inputs=gr.Textbox(
        lines=15,
        label="Enter Text to Summarize",
        placeholder="SUBREDDIT: r/explainlikeimfive\nTITLE: How does a microwave heat food? ...",
    ),
    outputs=gr.Textbox(
        lines=5,
        label="Generated Summary (TL;DR)",
    ),
    title="📝 RL-Finetuned GPT-2 Summarizer",
    description="This is a demo of a GPT-2 model fine-tuned with Reinforcement Learning (PPO) to summarize posts. Enter a prompt (like a Reddit post) and see the generated TL;DR.",
    allow_flagging="never",
    examples=[
        ["SUBREDDIT: r/explainlikeimfive\nTITLE: Why is the sky blue?\nPOST: I've always wondered why the sky is blue and not some other color like green or yellow. Can someone explain it in simple terms?\nTL;DR:"],
        ["SUBREDDIT: r/technology\nTITLE: New AI model achieves state-of-the-art results in protein folding.\nPOST: Scientists from a leading research lab have developed a new artificial intelligence model that can predict the 3D structure of proteins with unprecedented accuracy. This breakthrough could accelerate drug discovery and our understanding of various diseases. The model, called 'Structu-Net', leverages a novel deep learning architecture...\nTL;DR:"],
    ]
)

iface.launch()
