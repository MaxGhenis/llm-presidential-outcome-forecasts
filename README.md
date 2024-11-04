# LLM Presidential Policy Forecasts

This repository contains code and analysis examining how different large language models forecast policy outcomes under potential 2025 presidential administrations.

## Overview

Using narrative prompting techniques from [Cunningham et al. (2024)](https://arxiv.org/abs/2404.07396), this project:

- Compares predictions from GPT-4, GPT-4-mini, and Grok
- Analyzes forecasts for air quality, GDP, and poverty metrics
- Implements systematic prediction collection and analysis
- Examines differences in model behavior and accuracy

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up API keys:

```bash
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

3. Run analysis:

````bash

```bash
python -m llm_forecasts.main
````

## Repository Structure

- `paper/`: LaTeX paper and figures
- `src/`: Core Python implementation
- `data/`: Raw model outputs and processed results
- `notebooks/`: Analysis notebooks

## Results

Key findings:

- Significant predicted differences between candidates across metrics
- Systematic variations between models in effect sizes
- Evidence supporting narrative prompting effectiveness

## Citation

```bibtex
@misc{ghenis2024llm,
  author = {Ghenis, Max},
  title = {AI Model Policy Impact Forecasts: A Narrative Prompting Approach},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MaxGhenis/llm-forecasting}}
}
```

## License

MIT
