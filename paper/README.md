# Paper: AI Model Policy Impact Forecasts

## Structure

- `main.tex`: Main document file
- `sections/`: Individual section files
- `references.bib`: Bibliography file
- `Makefile`: Build commands

## Building

To compile the paper, run:

```bash
make
```

This will:

1. Run pdflatex
2. Process citations
3. Run pdflatex twice more to resolve references

## Sections

- Abstract
- Introduction
- Methodology
- Technical Implementation
- Results
- Discussion
- Conclusion

## Requirements

- A TeX distribution (e.g., TeXLive, MikTeX)
- make
- BibTeX
