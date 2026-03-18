# Multilingual Humor Generation System
### Automated Joke Generation using Mistral-7B + Prompt Engineering + Web Context Retrieval

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Mistral](https://img.shields.io/badge/Mistral-7B--Instruct-purple)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20ES%20%7C%20ZH-green)
![Internship](https://img.shields.io/badge/NIT%20Patna-Internship%20Project-orange)
![Human Eval](https://img.shields.io/badge/Human%20Eval-76%25%20Joke%20Recognition-brightgreen)

## What Does This Project Do?

This AI system automatically generates high-quality jokes from two types of inputs:

| Input Type | Example Input | Example Output |
|------------|---------------|----------------|
| News Headline | "Scientists discover water on Mars" | "Why did the Mars rover bring a towel? Because it heard they finally found water!" |
| Word Pair | "Python, bug" | "What do you call a snake that writes bad code? A Python with too many bugs!" |

Works in **3 languages simultaneously**: English, Spanish, and Chinese — without any fine-tuning.

---

## Technical Overview

| Detail | Value |
|--------|-------|
| Model | Mistral-7B-Instruct-v0.2 |
| Task | Multilingual humor generation from headlines & word pairs |
| Dataset | MWAHAHA benchmark (TSV format) |
| Languages | English (EN), Spanish (ES), Chinese (ZH) |
| Context Retrieval | DuckDuckGo Search (DDGS) |
| Quantization | 4-bit (bitsandbytes) for memory efficiency |
| Hardware | NVIDIA RTX 3090 (24GB VRAM) |
| Human Eval | 76% jokes recognized as genuinely funny |
| Constraint Satisfaction | 94% of constraints successfully enforced |
| Truncation Rate | Less than 5% of outputs truncated |

---

## Pipeline Architecture
```
INPUT (TSV File: headline or word pair)
    |
    v
+------------------------+
| Language Detection     |  Unicode range → ZH / Spanish markers → ES / Default → EN
+------------------------+
    |
    v
+------------------------+
| Web Context Retrieval  |  DuckDuckGo (DDGS) → 600-char context snippet
+------------------------+
    |
    v
+------------------------+
| Prompt Engineering     |  Language-specific instruction generation
| build_prompt()         |  EN / ES / ZH prompt templates
+------------------------+
    |
    v
+------------------------+
| LLM Inference          |  Mistral-7B-Instruct (4-bit quantized)
| generate_candidates()  |  n=5 candidates per input
+------------------------+
    |
    v
+------------------------+
| Constraint Enforcement |  Mandatory word inclusion, length check
| enforce_constraints()  |  50-120 character range
+------------------------+
    |
    v
+------------------------+
| Quality Filtering      |  Remove: truncated / banned phrases / emojis / hashtags
| is_valid_joke()        |  Minimum 7 words required
+------------------------+
    |
    v
+------------------------+
| Local Reranking        |  Heuristic scoring:
| local_rerank()         |  +2 setup present (?)
|                        |  +1 setup+punchline (\n)
|                        |  +2 optimal length (8-18 words)
|                        |  +2 wordplay / humor indicators
|                        |  -10 banned phrases
+------------------------+
    |         |
  PASS      FAIL → Retry (max 4 attempts) → Fallback joke
    |
    v
OUTPUT (TSV: id + best_joke)
```

---

## Project Structure
```
multilingual-humor-generation/
├── humor_pipeline.py         # Main pipeline
├── prompt_builder.py         # build_prompt() — multilingual prompts
├── context_retriever.py      # web_retrieve() — DuckDuckGo fetching
├── joke_validator.py         # is_valid_joke(), enforce_constraints()
├── reranker.py               # local_rerank() — heuristic scoring
├── sample_check.py           # Preview pipeline on first N samples
├── analysis/
│   ├── diversity_analysis.py
│   ├── ablation_study.py
│   └── visualizations.py
├── data/
│   ├── task-a-en[1].tsv
│   ├── task-a-es[1].tsv
│   └── task-a-zh[1].tsv
├── outputs/
│   ├── output_en.tsv
│   ├── output_es.tsv
│   └── output_zh.tsv
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Step 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/multilingual-humor-generation.git
cd multilingual-humor-generation
```

### Step 2 — Virtual Environment
```bash
conda create -n humor_gen python=3.9
conda activate humor_gen
```

### Step 3 — Install Dependencies
```bash
pip install transformers accelerate sentencepiece bitsandbytes
pip install ddgs tqdm pandas numpy torch
```

### Step 4 — HuggingFace Login
```bash
huggingface-cli login
# Enter your token
# Request access to: mistralai/Mistral-7B-Instruct-v0.2
```

---

## How to Run

### Quick test (first 5 samples)
```bash
python sample_check.py
```

### Full pipeline — English
```bash
python humor_pipeline.py --input data/task-a-en[1].tsv --output outputs/output_en.tsv
```

### Full pipeline — Spanish
```bash
python humor_pipeline.py --input data/task-a-es[1].tsv --output outputs/output_es.tsv
```

### Full pipeline — Chinese
```bash
python humor_pipeline.py --input data/task-a-zh[1].tsv --output outputs/output_zh.tsv
```

### Diversity analysis
```bash
python analysis/diversity_analysis.py
```

---

## Results

### Human Evaluation
| Metric | Score |
|--------|-------|
| Joke Recognition Rate (Human) | **76%** |
| Constraint Satisfaction | **94%** |
| Truncation Rate | < 5% |
| Clean Ending Rate | 100% |

### Multilingual Statistics
| Language | Jokes | Avg Words | Question-based (%) | Clean Endings (%) |
|----------|-------|-----------|-------------------|-------------------|
| English | 1,200 | 14.05 | 12.67% | 100% |
| Spanish | 1,200 | 17.65 | 17.75% | 100% |
| Chinese | 1,000 | 1.00 | 0.00% | 100% |

### Diversity Metrics
| Language | Distinct-1 | Distinct-2 | Embedding Diversity | Self-BLEU (inv.) |
|----------|------------|------------|--------------------|--------------------|
| English | 0.063 | 0.129 | **0.807** | 0.059 |
| Spanish | **0.159** | **0.327** | 0.551 | **0.955** |
| Chinese | 0.010 | 0.000 | 0.318 | 0.822 |

### vs Existing Systems
| System | Approach | Human Eval |
|--------|----------|------------|
| Witscript | NLP keyword + fine-tuned LM | 40%+ joke recognition |
| LoL Framework | Structured thought leaps + RL | Enhanced creative capability |
| **Our System** | **Mistral-7B + multilingual prompts** | **76% joke recognition** |

---

## Prompt Engineering Strategy
```python
# English
"Headline: {headline}\nMake one short joke that plays with the words
 or meaning in the headline. Keep it under 100 characters."

# Spanish
"Título de la noticia: {headline}\nEscribe un chiste corto en español.
 Utiliza un juego de palabras relacionado con el título."

# Chinese
"新闻标题: {headline}\n请用中文写一个简短的笑话。"
```

---

## Heuristic Scoring Function
```python
def score_joke(joke):
    score = 0
    if "?" in joke: score += 2
    if ":" in joke and "?" in joke: score += 3
    if "\n" in joke: score += 1
    words = len(joke.split())
    if 8 <= words <= 18: score += 2
    elif 5 <= words <= 25: score += 1
    if any(w in joke.lower() for w in ["pun", "play on words", "double meaning"]):
        score += 2
    if any(banned in joke.lower() for banned in banned_phrases):
        score -= 10
    return score
```

---

## Sample Generated Jokes

| Input | Generated Joke |
|-------|----------------|
| Headline: "Scientists discover water on Mars" | "Why did the Mars rover bring a towel? Because it heard they finally found water!" |
| Words: "Python, bug" | "What do you call a snake that writes bad code? A Python with too many bugs!" |
| Headline: "New AI passes Turing test" | "The AI was so good at pretending to be human, it asked for a raise and complained about Mondays." |
| Words: "Cloud, server" | "Why was the cloud server always calm? Because it never lost its files — they were just in another data center!" |

---

## Limitations

- Chinese generation quality lower than English/Spanish
- No fine-tuning — all multilingual behavior from prompt conditioning only
- Humor is subjective — BLEU/ROUGE poorly capture joke quality
- Web context retrieval adds ~1-2 seconds latency per joke
- Culturally-specific humor may not transfer across languages

---

## Future Work

- [ ] Reinforcement learning from human feedback (RLHF)
- [ ] Web-based demo interface for interactive testing
- [ ] Extended multilingual support (French, Hindi, Japanese)
- [ ] Multimodal humor — image + text joke generation
- [ ] Personalized humor adaptation per user
- [ ] Real-time generation with <100ms latency

---

## Internship Details

- **Institute:** National Institute of Technology (NIT), Patna
- **Department:** Computer Science & Engineering
- **Guide:** Dr. Jyoti Prakash Singh, Associate Professor
- **Student:** Uttam Kumar (24A12res1147)
- **Date:** 29 December 2025

---

## Tech Stack

![Mistral](https://img.shields.io/badge/Mistral-7B--Instruct-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![DuckDuckGo](https://img.shields.io/badge/DDGS-Web%20Search-blue)
![4bit](https://img.shields.io/badge/4bit-Quantization-green)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with Mistral-7B + HuggingFace Transformers | Dataset: MWAHAHA Benchmark | NIT Patna Internship 2025*
