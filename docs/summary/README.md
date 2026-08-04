# Codebase Summary Report

`pipeline_report.tex` / `pipeline_report.pdf` — a self-contained LaTeX report
explaining the full pop2vec LLM pipeline (data prep → pretraining (MLM/AR) →
static-embedding and finetuned prediction routes), with TikZ diagrams in the
style of the BERT / "Attention is All You Need" papers.

Rebuild with:

```bash
pdflatex pipeline_report.tex   # run twice for TOC / cross-references
```

No external figures or bibliography files are needed; everything is drawn with
TikZ inside the single `.tex` file.
