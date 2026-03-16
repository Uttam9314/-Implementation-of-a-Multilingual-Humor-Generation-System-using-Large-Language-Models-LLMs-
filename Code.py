# -*- coding: utf-8 -*-

!pip install -q transformers accelerate sentencepiece ddgs tqdm

import csv
import time
from tqdm import tqdm
from ddgs import DDGS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)


model.eval()

def web_retrieve(query, max_results=5):
    snippets = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            snippets.append(r.get("body", ""))
    return " ".join(snippets)[:600]

def detect_language(text):
    # Chinese
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "zh"

    # Spanish (basic but effective)
    spanish_markers = ["¿", "¡", "ñ", "á", "é", "í", "ó", "ú"]
    if any(m in text.lower() for m in spanish_markers):
        return "es"

    # Default
    return "en"

def build_prompt(mode, headline=None, word1=None, word2=None):
    base_text = headline if headline else f"{word1} {word2}"
    lang = detect_language(base_text)

    # ---------- CHINESE ----------
    if lang == "zh":
        if mode == "headline":
            return f"新闻标题：{headline}\n请用中文写一个简短的笑话。"
        else:
            return f"请用中文写一个简短的笑话，必须包含这两个词：{word1}，{word2}。"

    # ---------- SPANISH ----------
    if lang == "es":
        if mode == "headline":
            return f"Título de la noticia: {headline}\nEscribe un chiste corto en español."
        else:
            return f"Escribe un chiste corto en español que incluya las palabras {word1} y {word2}."

    # ---------- ENGLISH ----------
    if mode == "headline":
        return f"Headline: {headline}\nMake one short joke."
    else:
        return f"Make one short joke using the words {word1} and {word2}."

def generate_joke(prompt, max_new_tokens=90):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.95,
            top_p=0.9,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split(prompt)[-1].strip()

def trim_joke(joke, max_len=120):
    return joke.strip()[:max_len]

def generate_candidates(prompt, n=5):
    jokes = []
    for _ in range(n):
        j = generate_joke(prompt)
        if j and j not in jokes:
            jokes.append(j)
    return jokes

def enforce_constraints(joke, mode, word1=None, word2=None):
    if mode == "word":
        if word1.lower() not in joke.lower():
            joke += f" {word1}"
        if word2.lower() not in joke.lower():
            joke += f" {word2}"
    return joke

def local_rerank(jokes):
    def score(j):
        s = 0
        if "?" in j:
            s += 2          # jokes with setup
        if "\n" in j:
            s += 1          # setup + punchline
        if ":" in j and "?" not in j:
            s -= 1          # slogans
        l = len(j.split())
        if 8 <= l <= 18:
            s += 1
        return s

    return max(jokes, key=score)

def generate_clean_joke(prompt, max_tries=4):
    for _ in range(max_tries):
        jokes = generate_candidates(prompt, n=5)

        # filter garbage
        jokes = [j for j in jokes if is_valid_joke(j)]

        if not jokes:
            continue

        best = local_rerank(jokes)

        # truncation check
        if looks_truncated(best):
            continue

        return trim_joke(best)

    # last-resort fallback (safe)
    return "Mars found water and immediately raised the rent."

def looks_truncated(joke):
    if len(joke) < 10:
        return True

    # ends mid-word
    if joke[-1].isalnum():
        return True

    # bad endings
    if joke.strip().endswith(("p", "kn", "know", "giv", "alread")):
        return True

    return False

def is_valid_joke(joke):
    joke_l = joke.lower()

    banned = [
        "just a joke", "just the joke", "no setup", "no set up",
        "just a short", "short, punchy", "write", "context",
        "headline", "note:", "make one"
    ]

    if any(b in joke_l for b in banned):
        return False

    if "#" in joke or any(c in joke for c in "😂🤣😄🤪🚀🥳"):
        return False

    if len(joke.split()) < 7:
        return False

    return True

def load_input(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def run_pipeline(input_file, output_file):
    rows = load_input(input_file)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"], delimiter="\t")
        writer.writeheader()

        for i, row in enumerate(tqdm(rows), 1):

            # decide task type
            if row["word1"] != "-" and row["word2"] != "-":
                prompt = build_prompt(
                    mode="word",
                    word1=row["word1"],
                    word2=row["word2"]
                )
            else:
                prompt = build_prompt(
                    mode="headline",
                    headline=row["headline"]
                )

            # FINAL clean generation
            best_joke = generate_clean_joke(prompt)

            writer.writerow({
                "id": row["id"],
                "text": best_joke
            })

            # show first few for confidence
            if i <= 3:
                print(f"\nWritten {i}: {best_joke}")

    print(f"\n✅ Done. Output saved to{output_file}")

def load_input(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)



def sample_check(input_file, n_samples=5):
    rows = load_input(input_file)

    print(f"\n🔍 Previewing first {n_samples} samples\n")

    for i, row in enumerate(rows[:n_samples], 1):
        print("=" * 80)
        print(f"SAMPLE {i}")

        # Decide task type
        if row["word1"] != "-" and row["word2"] != "-":
            print("MODE: WORD-INCLUSION")
            print("Words:", row["word1"], ",", row["word2"])

            prompt = build_prompt(
                mode="word",
                word1=row["word1"],
                word2=row["word2"]
            )
        else:
            print("MODE: HEADLINE")
            print("Headline:", row["headline"])

            prompt = build_prompt(
                mode="headline",
                headline=row["headline"]
            )

        # Generate final clean joke
        joke = generate_clean_joke(prompt)

        print("\nFINAL JOKE:")
        print(joke)

    print("\n✅ Sample check done")

import csv
sample_check(
    input_file="task-a-en.tsv",
    n_samples=5
)

