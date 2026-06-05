# DL4NLP — LiU Course Assignments

Solutions for the four assignments in [LiU's Deep Learning for NLP](https://liu-nlp.ai/dl4nlp/) course.

Author: Wei Shi

> **AI usage:** Parts of this submission were prepared with help from an
> AI coding assistant. See [`AI_USAGE.md`](./AI_USAGE.md) for a detailed
> statement of how AI was — and was not — used.

## Layout

```
assignment1/   Byte-level language model from scratch (Tiny Shakespeare)
assignment2/   Transformer-based language model
assignment3/   Supervised fine-tuning + LoRA on SmolLM2-135M
assignment4/   Retrieval-Augmented Generation over PubMedQA
```

Each folder contains:

| File | Purpose |
|---|---|
| `A{n}_solution.ipynb` | Filled-in notebook with executed outputs (the deliverable) |
| `A{n}_skeleton.{ipynb,py}` | Original course skeleton, untouched |
| `trained_model/` | (A1, A2) Saved model weights from the run |
| `tokenizer.pkl` | (A1) Trained byte-pair tokenizer |

## Reproducibility

The notebooks are self-contained: each downloads its dataset (Tiny Shakespeare, SmolTalk, PubMedQA) on first run.

Generated artefacts that are **not** checked in (regenerable from the notebooks):

- `assignment1/train.txt`, `assignment1/val.txt` — Tiny Shakespeare splits (downloaded by the notebook)
- `assignment3/out_full_sft/`, `out_lora_sft/`, `out_pretrained_eval/` — Trainer working dirs
- `assignment4/data/ori_pqal.json` — PubMedQA dump (downloaded by the notebook)
- `assignment4/chroma_pubmedqa/` — Chroma vector store (rebuilt on first run)

## Environment

- Python 3.13
- PyTorch 2.11 (CUDA, MPS, or CPU — notebooks auto-select)
- HuggingFace `transformers`, `datasets`, `evaluate`
- (A3) `peft` not required — LoRA is implemented manually
- (A4) `langchain`, `langchain-huggingface`, `langchain-chroma`,
  `sentence-transformers`, `chromadb`

Dependencies are installed by the first `%pip install` cell in each notebook.

## Notes per assignment

### A1 — Byte-level language model
Train a small byte-level language model on Tiny Shakespeare from scratch.

### A2 — Transformer LM
Continuation of A1, larger transformer model.

### A3 — SFT + LoRA
Full supervised fine-tuning vs. LoRA on SmolLM2-135M with SmolTalk.
LoRA is implemented manually (no `peft`) — see `LoRALayer` and
`extract_lora_targets` in the notebook. Runs on CUDA, Apple-Silicon MPS,
and CPU; bf16 is enabled only on CUDA.

### A4 — RAG over PubMedQA
LangChain LCEL pipeline: MiniLM embeddings → Chroma (cosine) →
Qwen2.5-0.5B-Instruct. Eval on 50 questions:
RAG ≈ 64% accuracy / 0.61 macro-F1 vs. LM-only ≈ 44% / 0.43;
retrieval recall@k=3 ≈ 98%.
