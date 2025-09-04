import torch
from flask import Flask, request, jsonify, render_template

# Make sure the mingpt folder is accessible
from mingpt.model import GPT
from mingpt.char_tokenizer import CharTokenizer

# --- DATASET CLASS (needed to build the tokenizer) ---
import datasets
from torch.utils.data import Dataset

class StoryDataset(Dataset):
    """
    Minimal StoryDataset class to rebuild the tokenizer from the training data sample.
    """
    def __init__(self, split="train"):
        eot = "⏎"
        def chunk_examples(examples):
            chunks = [(eot + text + eot) for text in examples['text'] if len(text) > 0]
            return {"content": chunks}

        # Load a small part of the dataset, which is enough to build the tokenizer
        # This MUST match the number of samples used in the original training script (10000)
        print("Loading a small sample of TinyStories to build tokenizer...")
        dataset = datasets.load_dataset("roneneldan/TinyStories", split=f"{split}[:10000]")
        self.dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)

        # Build tokenizer from the dataset content
        print("Building tokenizer...")
        token_text_sample = "".join(row["content"] for row in self.dataset)
        self.tokenizer = CharTokenizer(token_text_sample)
        print("Tokenizer built successfully.")

    def __len__(self):
        # Not needed for tokenizer, but required by the abstract class
        return len(self.dataset)

    def __getitem__(self, idx):
        # Not needed for tokenizer, but required by the abstract class
        return self.dataset[idx]["content"]

# --- CONFIGURATION ---
PRETRAINED_MODEL_PATH = "story_gpt_pretrained.pt"
RL_MODEL_PATH = "story_gpt_rl.pt"
DEVICE = "cpu" # Use 'cuda' if you have a GPU
BLOCK_SIZE = 64 # This must match the block_size used during training

# --- LOAD TOKENIZER (do this once on startup) ---
# This will download a small part of the dataset to ensure the tokenizer is identical
train_ds = StoryDataset(split='train')
tokenizer = train_ds.tokenizer
vocab_size = tokenizer.vocab_size

# --- LOAD MODELS (do this once on startup) ---
def load_model(model_path):
    """Initializes a GPT model with the correct config and loads the saved weights."""
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = vocab_size
    model_config.block_size = BLOCK_SIZE
    
    model = GPT(model_config)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    except FileNotFoundError:
        print(f"--- WARNING: Model file not found at {model_path}. This endpoint will fail. ---")
        return None
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    return model

print("Loading models...")
pretrained_model = load_model(PRETRAINED_MODEL_PATH)
rl_model = load_model(RL_MODEL_PATH)
print("Models loaded.")

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- WEB APP ROUTE ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- API ENDPOINT ---
@app.route('/generate', methods=['POST'])
def generate():
    """Handles the text generation request."""
    if not pretrained_model or not rl_model:
        return jsonify({'error': 'One or more models are not loaded. Check server logs.'}), 500

    data = request.get_json()
    prompt_text = data.get('prompt', 'Once upon a time')
    max_tokens = int(data.get('max_tokens', 100))
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 30))
    
    # Prepare the prompt tensor
    eot = "⏎"
    full_prompt = eot + prompt_text
    context = torch.tensor(tokenizer(full_prompt)[None, ...], device=DEVICE)

    # Generate from both models
    with torch.no_grad():
        pretrained_completion_tok = pretrained_model.generate(
            context, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_k=top_k
        )[0]
        rl_completion_tok = rl_model.generate(
            context, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_k=top_k
        )[0]
    
    # Decode the generated tokens back to text
    # We strip the initial prompt and the EOT token for a cleaner output
    pretrained_text = tokenizer.decode(pretrained_completion_tok).replace(full_prompt, '', 1)
    rl_text = tokenizer.decode(rl_completion_tok).replace(full_prompt, '', 1)
    
    return jsonify({
        'pretrained_output': pretrained_text.strip(),
        'rl_output': rl_text.strip()
    })

if __name__ == '__main__':
    # Make sure you have a 'templates' folder with 'index.html' in it.
    app.run(host='0.0.0.0', port=5000)
