# DL4NLP — LiU Course Assignments

Solutions for the four assignments in [LiU's Deep Learning for NLP](https://liu-nlp.ai/dl4nlp/) course.

Author: Wei Shi

> **AI usage:** Parts of this submission were prepared with help from an
> AI coding assistant. See [`AI_USAGE.md`](./AI_USAGE.md) for a detailed
> statement of how AI was — and was not — used.

## Layout

```
assignment1/   Word-level LSTM language model from scratch (Simple English Wikipedia)
assignment2/   Transformer-based language model (OLMo-2 style)
assignment3/   Supervised fine-tuning + LoRA on SmolLM2-135M
assignment4/   Retrieval-Augmented Generation over PubMedQA
```

Each folder contains:

| File | Purpose |
|---|---|
| `A{n}_solution.ipynb` | Filled-in notebook with executed outputs (the deliverable) |
| `A{n}_skeleton.{ipynb,py}` | Original course skeleton (A1–A3; A4 scaffolding was AI-generated) |
| `trained_model/` | (A1, A2) Saved model weights from the run |
| `tokenizer.pkl` | (A1) Trained word-level tokenizer (NLTK + 4 special tokens) |

## Reproducibility

A1 reads pre-existing `train.txt` / `val.txt` files (Simple English Wikipedia paragraphs, provided with the course). A3 and A4 download their datasets (SmolTalk, PubMedQA) on first run.

Generated artefacts that are **not** checked in (regenerable from the notebooks):

- `assignment3/out_full_sft/`, `out_lora_sft/`, `out_pretrained_eval/` — Trainer working dirs
- `assignment4/data/ori_pqal.json` — PubMedQA dump (downloaded by the notebook)
- `assignment4/chroma_pubmedqa/` — Chroma vector store (rebuilt on first run)

Required files that must be present:

- `assignment1/train.txt`, `assignment1/val.txt` — Simple English Wikipedia text (provided with the course assignment, not downloaded by the notebook)

## Environment

- Python 3.13
- PyTorch 2.11 (CUDA, MPS, or CPU — notebooks auto-select)
- HuggingFace `transformers`, `datasets`, `evaluate`
- (A3) `peft` not required — LoRA is implemented manually
- (A4) `langchain`, `langchain-huggingface`, `langchain-chroma`,
  `sentence-transformers`, `chromadb`

A3 and A4 install extra dependencies via a `%pip install` cell.

## Notes per assignment

### A1 — Word-level LSTM language model
Train a small word-level LSTM language model from scratch on a Simple English Wikipedia paragraph corpus. Tokenizer is NLTK `word_tokenize` + lowercasing, with 4 reserved special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).

### A2 — Transformer LM
Continuation of A1 (same data pipeline and tokenizer), with the LSTM replaced by a small OLMo-2-style decoder (RMSNorm, RoPE, SwiGLU MLP, QK-Norm, pre-norm residual blocks). Also includes a side-by-side qualitative comparison with the pre-trained OLMo-2 1B.

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
