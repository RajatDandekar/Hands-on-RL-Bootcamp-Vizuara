import os, logging, datetime, json

from dotenv import load_dotenv
from rich.traceback import install

load_dotenv()
install()

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

# IMPORTANT: utils live under src/
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)
from src.models.model import AgenticRAGModel

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in prod, restrict to your UI origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str

# --- setup logging & config ---
setup_logging()
# use the same config path train.py uses
config = load_config("src/config/config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# point to your trained LoRA adapter (adjust this to your actual best step)
# e.g., checkpoints/debug/2025-09-22/step-0005
checkpoint_path = os.getenv(
    "LORA_ADAPTER_PATH",
    "checkpoints/debug/2025-09-22/step-0005"
)

logging.info(f"Loading base model from {config.model.name}")
torch_dtype = getattr(torch, config.model.torch_dtype)

base_model = AutoModelForCausalLM.from_pretrained(
    config.model.name,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    config.model.name,
    padding_side="left",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id

# --- LoRA ---
# your YAML uses `lora: ...`
lora_cfg = LoraConfig(
    r=config.lora.r,
    lora_alpha=config.lora.lora_alpha,
    target_modules=config.lora.target_modules,
    lora_dropout=config.lora.lora_dropout,
    bias=config.lora.bias,
    task_type=config.lora.task_type,
)

if checkpoint_path and os.path.isdir(checkpoint_path):
    logging.info(f"Loading LoRA weights from {checkpoint_path}")
    base_model = PeftModel.from_pretrained(base_model, checkpoint_path)
else:
    logging.warning(f"LoRA adapter path not found: {checkpoint_path}. Serving base model only.")

base_model = base_model.to(device)
model = AgenticRAGModel(base_model, tokenizer)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat/")
async def chat(query: Query):
    inputs = tokenizer([query.text], return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        max_length_for_gather=4096,
        do_sample=False,
        temperature=0.8,
    )
    gen = output_ids[0][len(input_ids[0]):]
    text = tokenizer.decode(gen, skip_special_tokens=True, spaces_between_special_tokens=False)
    return {"result": text}

if __name__ == "__main__":
    # bind to 0.0.0.0 so Runpod can expose it
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "12333")))
