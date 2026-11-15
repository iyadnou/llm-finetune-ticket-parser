# Project 1 — Fine-tune a small LLM to parse support tickets

**Goal:** turn messy customer messages into clean JSON.

```
IN : "hey team, PhotoPro crashes every time I open it. URGENT!! - Amina"
OUT: {"category": "bug", "urgency": "high", "product": "PhotoPro",
      "order_id": null, "customer_name": "Amina"}
```

We start from **Qwen2.5-0.5B-Instruct** (a 500M-parameter chat model). It
already speaks English and knows JSON, but it doesn't know *our* categories
or rules. Fine-tuning teaches it our task from 1,000 examples.

## Results

Scores on the 100 test tickets (never seen during training), with the
settings in `lora_config.yaml` (400 steps, about 0.6% of weights trained):

| Metric | Base model | Fine-tuned (LoRA) |
|---|---|---|
| Valid JSON | 100% | 100% |
| category | 43% | **100%** |
| urgency | 38% | **100%** |
| product | 89% | **100%** |
| order_id | 95% | **100%** |
| customer_name | 87% | **100%** |
| **All fields correct** | **13%** | **100%** |

**Caveat:** the test tickets come from the same templates as the training
tickets, so this task is easy for the fine-tuned model. On real, messy
customer emails the score would be lower. Measuring on real data is the
next step (Project 2).

---

## The 5 ideas you need first

1. **Pretrained model.** Someone already spent millions training a model on
   the internet. You never start from zero; you *adapt* an existing model.
2. **Fine-tuning.** You keep training that model on your own
   (input → desired output) examples, so it gets very good at one narrow job.
3. **Loss.** A number measuring "how surprised the model is by the correct
   answer". Training pushes it down. You watch two losses:
   - *train loss*: on data it learns from
   - *val loss*: on data it never learns from. If train goes down but val
     goes **up**, the model is memorizing (**overfitting**).
4. **LoRA.** Updating all 500M weights is expensive. LoRA freezes the model
   and trains small extra matrices ("adapters") instead. Here that's about
   **0.6%** of the parameters (2.9M), and the result is a small file
   you load on top of the base model.
5. **Evaluation.** Loss is for training. To know if the model is *useful*,
   you measure the real task (here: % of tickets fully correct) **before and
   after**.

---

## Setup (already done once)

```bash
cd ~/VSCProjects/Fine-tune
source .venv/bin/activate        # do this in every new terminal
```

The environment uses Python 3.11 + `mlx-lm` (Apple's library that trains on
your M2's GPU). Reinstall later with `pip install -r requirements.txt`.

## Step 1 — Look at the data

```bash
python 01_make_data.py
head -n 3 data/train.jsonl
```

Read [01_make_data.py](01_make_data.py). Notice the **train / valid / test**
split. The test set is never used in training; it's your honest exam.

## Step 2 — Measure the base model (before)

```bash
python evaluate.py
```

Write down the numbers. On a 10-ticket check, the base model got about
**30% all-correct**. It often confuses `feature_request` with `bug` and
guesses urgency.

## Step 3 — Fine-tune (about 5–10 minutes)

```bash
mlx_lm.lora -c lora_config.yaml
```

Read [lora_config.yaml](lora_config.yaml) while it runs. Watch the output:
- `train_loss` should drop fast (from about 1.1 toward about 0.05)
- `val` loss appears every 50 steps and should drop too

Close other heavy apps. You have 8 GB of RAM.

## Step 4 — Measure again (after)

```bash
python evaluate.py --adapter adapters
```

Compare with Step 2. That difference is what fine-tuning did.

## Step 5 — Play with it

```bash
mlx_lm.generate --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters \
  --system-prompt "$(python -c 'import importlib;print(importlib.import_module("01_make_data").SYSTEM_PROMPT)')" \
  --prompt "yo my MailMate account is locked, need it back NOW. Karim"
```

Try messages that look **different** from the templates (other languages,
slang, two problems at once). Where it breaks shows you what your training
data was missing. That lesson matters most in real projects.

Optional: merge adapters into a standalone model:
`mlx_lm.fuse --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters --save-path fused_model`

---

## Experiments (do at least 3, one change at a time)

Change one setting, retrain into a new folder (`--adapter-path adapters_x`),
evaluate, and note the result in a small table.

| Experiment | What you'll learn |
|---|---|
| `--iters 50` vs `400` vs `1500` | under-training vs overfitting (watch val loss) |
| Train on only 100 examples (edit the split) | how much data you really need |
| `rank: 2` vs `rank: 32` | adapter capacity |
| `learning_rate: 1e-3` or `1e-6` | too high = unstable, too low = no learning |
| `mask_prompt: false` | why we train only on the answer |
| Remove the system prompt from the data | fine-tuning can replace long prompts |

---

## What this project skipped (you'll need it in Project 2)

- **Real data:** collecting, cleaning, deduplicating, and labelling it. In companies,
  this is about 70% of the work.
- **Hugging Face stack:** `transformers` + `peft` + `trl` (SFTTrainer) on an
  NVIDIA GPU (Colab, Kaggle, RunPod). This is the industry standard. MLX is
  the Mac equivalent, and the concepts are identical.
- **QLoRA:** LoRA on a 4-bit quantized model, to fine-tune 7B–8B models on
  one GPU.
- **Proper evaluation:** held-out *real* data, comparison against a
  prompted big model (is fine-tuning even worth it?), LLM-as-judge for
  free-text outputs.
- **Experiment tracking:** Weights & Biases (`--report-to wandb`).
- **Serving:** vLLM / Ollama, latency and cost.
- **Preference tuning:** DPO, for teaching style and "which answer is better".

### Ideas for Project 2 (company-style)

- Fine-tune Llama-3.1-8B / Qwen2.5-7B with QLoRA on a real public dataset
  (e.g. customer support, legal clause classification, text-to-SQL) on a
  cloud GPU, and beat a prompted baseline. Write it up with metrics.
- Domain assistant: fine-tune on a company's docs/Q&A, then compare
  against RAG (retrieval). Knowing *when not to fine-tune* is a senior skill.
