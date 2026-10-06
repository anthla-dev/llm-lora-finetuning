# LLM Fine-Tuning with LoRA

## What it does
Fine-tunes a Qwen2.5-0.5B language model using LoRA (Low-Rank Adaptation) on the
Databricks Dolly 15k instruction dataset. Training metrics are tracked in real time
using Weights & Biases.

## Techniques
- LoRA fine-tuning via Hugging Face PEFT (rank=8, alpha=16)
- Target modules: q_proj, v_proj
- Gradient accumulation (4 steps)
- Real-time loss and eval tracking via Weights & Biases
- CPU and CUDA compatible

## Results
- Trained on 450 samples (500 shuffled, 10% held out for eval)
- Evaluated every 50 steps

## Run it
Set environment variables:
```bash
export WANDB_API_KEY=your_key
export HF_TOKEN=your_token
export HF_DATASET=databricks/databricks-dolly-15k
export HF_MODEL=Qwen/Qwen2.5-0.5B
```

Then run:
```bash
pip install torch transformers datasets peft wandb
python train.py
```

## Skills demonstrated
Python, PyTorch, Hugging Face Transformers, PEFT, LoRA, Weights & Biases, LLM fine-tuning
