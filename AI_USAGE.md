# AI Usage Statement

This submission was completed with the assistance of an AI coding
assistant (Anthropic's Claude, accessed through the Claude Code CLI).
In the interest of academic transparency, this document describes
how AI was — and was not — used.

## How AI was used

- **Skeleton scaffolding.** For Assignment 4, where no course-provided
  skeleton was available, the AI generated the initial notebook
  scaffolding (markdown headers, library imports, configuration cells,
  task descriptions, and `TODO` placeholders). The structure mirrors
  the style of the official A1/A2/A3 skeletons.
- **Teaching and explanation.** The AI was used as an interactive
  tutor — to explain concepts (LoRA initialisation, ChatML format,
  LangChain LCEL, cosine vs L2 similarity, retrieval evaluation
  pitfalls), to discuss design trade-offs, and to debug environment
  issues (Apple-Silicon MPS device handling, HuggingFace pipeline
  construction, conda vs venv environment selection in VS Code).
- **Boilerplate and idiomatic code.** Repetitive or library-specific
  boilerplate (e.g. Trainer arguments, prompt templates, plotting
  helpers) was drafted with AI assistance and then reviewed and
  adjusted to match the course requirements.
- **Code review and sanity checks.** The AI was used as a second pair
  of eyes to flag inconsistencies, suspicious results, or potential
  bugs in the executed notebooks.

## How AI was *not* used

- The **conceptual decisions** for each assignment (which model, which
  retrieval strategy, which evaluation metric, how to interpret the
  results) were made by me.
- The **executed results** in each `*_solution.ipynb` come from running
  the code on my own hardware. No outputs were fabricated by the AI.
- The **course assignments themselves** (problem statements, dataset
  choice, grading criteria) are entirely the work of the LiU course
  staff; the AI did not author them.

## Reproducibility

A3 and A4 download their datasets on first run; A1 and A2 read the
course-provided `train.txt` / `val.txt`. The executed cells in each
`*_solution.ipynb` reflect actual runs on the listed hardware (CUDA /
Apple-Silicon MPS / CPU, auto-selected). Re-executing the notebooks
should produce numerically similar results modulo seed-dependent
variation.

## Tools

- **Anthropic Claude** (Claude Code CLI) — code authoring, debugging,
  explanation.
- **Standard course tooling** — PyTorch, HuggingFace `transformers`
  / `datasets`, LangChain, Chroma, scikit-learn.

If anything in this submission appears unclear or inconsistent with
the spirit of the course's AI-use policy, please contact me and I
will be happy to clarify.
