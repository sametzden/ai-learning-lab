# ai-learning-lab

Things I build while learning machine learning and LLM tooling. Each folder is a separate, self-contained piece of work; some are tutorials I followed, some are my own experiments.

| Folder | What it is |
|---|---|
| [`pytorch-fundamentals/`](pytorch-fundamentals) | First steps with PyTorch tensors. |
| [`change-detection/`](change-detection) | Siamese U-Net written from scratch in PyTorch for building change detection on satellite image pairs (LEVIR-CD). F1 0.79 / IoU 0.65 on the test set after 50 epochs. |
| [`shakespeare-gpt/`](shakespeare-gpt) | Character-level GPT (decoder-only Transformer) written from scratch in PyTorch, trained on Tiny Shakespeare. |
| [`llm-projects/`](llm-projects) | LangChain, RAG and embedding experiments. |
| [`tutorials/langgraph/`](tutorials/langgraph) | LangGraph tutorial notebooks. |
| [`tutorials/agentic-ai/`](tutorials/agentic-ai) | Agentic AI tutorial notebooks and small apps. |

## Notes

- Datasets and trained weights are not committed. The change detection notebook expects LEVIR-CD under `change-detection/data-change-detection/` with `train/`, `val/` and `test/` folders, each containing `A/`, `B/` and `label/`.
- `shakespeare-gpt/`, `llm-projects/` and the two `tutorials/` folders were imported from their own repositories with their commit history.
