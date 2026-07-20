"""RAGAS evaluation for the DGCA RAG system.

Implements Faithfulness, Context Recall, and Context Precision using
direct synchronous LLM calls with the Ollama qwen2.5:7b evaluator.

Usage:
    python evaluate_rag.py                    # evaluate with all 51 samples
    python evaluate_rag.py --samples 5       # quick test with 5 samples

Requires:
    pip install faiss-cpu openai langchain-ollama langchain-openai
    Ollama running locally with qwen2.5:7b pulled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "eval_results"
VOTING_RUNS = 3


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

class LLMCaller:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def call(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=10,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise


def build_llm_caller(evaluator_choice: str = "auto") -> LLMCaller:
    if evaluator_choice in ("auto", "azure"):
        if (
            os.getenv("AZURE_OPENAI_API_KEY")
            and os.getenv("AZURE_OPENAI_ENDPOINT")
            and os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        ):
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
            deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
            print("  Using Azure OpenAI as evaluator")
            return LLMCaller(client, deployment)
        elif evaluator_choice == "azure":
            print("Error: Azure OpenAI credentials not found.", file=sys.stderr)
            sys.exit(1)

    if evaluator_choice in ("auto", "ollama"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            print("  Using Ollama qwen2.5:7b as evaluator")
            return LLMCaller(client, "qwen2.5:7b")
        except Exception:
            pass

    print("[ERROR] No evaluator LLM available.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Ask the LLM for a score from 1 to 5
# ---------------------------------------------------------------------------

SYSTEM_SCORE = (
    "You are a precise evaluator. Rate the quality from 1 to 5.\n"
    "1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent.\n"
    "Reply with ONLY a single digit (1, 2, 3, 4, or 5). Nothing else."
)


def _ask_score_1_5(llm: LLMCaller, prompt: str) -> float | None:
    """Ask the LLM to rate from 1 to 5. Returns 0.0 to 1.0."""
    try:
        raw = llm.call(SYSTEM_SCORE, prompt).strip()
        digits = re.findall(r'[1-5]', raw)
        if digits:
            return (int(digits[0]) - 1) / 4.0  # map 1-5 to 0.0-1.0
        return None
    except Exception as e:
        print(f"      [WARN] score call failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Faithfulness — per-chunk 1-5 scoring
# ---------------------------------------------------------------------------

def metric_faithfulness(
    llm: LLMCaller,
    question: str,
    response: str,
    reference: str,
    chunks: list[str],
) -> float | None:
    if not chunks:
        return 0.0

    context_block = "\n\n".join(f"[Chunk {i+1}] {c[:500]}" for i, c in enumerate(chunks))
    prompt = (
        "How faithful is the generated response to the retrieved context?\n"
        "Faithfulness means: every claim in the response is supported by the context.\n"
        "Score 5 if all claims are supported. Score 1 if none are.\n\n"
        f"Context:\n{context_block[:1500]}\n\n"
        f"Generated Response:\n{response[:500]}\n\n"
        "Score (1-5):"
    )
    return _ask_score_1_5(llm, prompt)


# ---------------------------------------------------------------------------
# Context Recall — per-sentence 1-5 scoring
# ---------------------------------------------------------------------------

def metric_context_recall(
    llm: LLMCaller,
    question: str,
    response: str,
    reference: str,
    chunks: list[str],
) -> float | None:
    if not reference:
        return 0.0

    context_block = "\n\n".join(f"[Chunk {i+1}] {c[:500]}" for i, c in enumerate(chunks))
    if not context_block:
        return 0.0

    prompt = (
        "How well does the retrieved context cover the reference answer?\n"
        "Good recall means: the context contains enough info to answer the question as well as the reference.\n"
        "Score 5 if the context fully covers the reference. Score 1 if it covers nothing.\n\n"
        f"Reference Answer:\n{reference[:500]}\n\n"
        f"Retrieved Context:\n{context_block[:1500]}\n\n"
        "Score (1-5):"
    )
    return _ask_score_1_5(llm, prompt)


# ---------------------------------------------------------------------------
# Context Precision — per-chunk 1-5 scoring
# ---------------------------------------------------------------------------

def metric_context_precision(
    llm: LLMCaller,
    question: str,
    response: str,
    reference: str,
    chunks: list[str],
) -> float | None:
    if not chunks:
        return 0.0

    chunks_text = "\n\n".join(f"[Chunk {i+1}] {c[:400]}" for i, c in enumerate(chunks))
    prompt = (
        "How precise is the retrieved context? Precision means most chunks are relevant to the question.\n"
        "Score 5 if all chunks are highly relevant. Score 1 if most are irrelevant.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Chunks:\n{chunks_text}\n\n"
        "Score (1-5):"
    )
    return _ask_score_1_5(llm, prompt)


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

def _median_score(scores: list[float | None]) -> float | None:
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return statistics.median(valid)


def score_with_voting(
    llm: LLMCaller,
    metric_fn,
    metric_name: str,
    question: str,
    response: str,
    reference: str,
    chunks: list[str],
    runs: int = VOTING_RUNS,
) -> float | None:
    scores = []
    for run_idx in range(runs):
        try:
            s = metric_fn(llm, question, response, reference, chunks)
            scores.append(s)
        except Exception as e:
            print(f"      [WARN] Run {run_idx + 1} failed: {e}", file=sys.stderr)
            scores.append(None)

    return _median_score(scores)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_query(question: str) -> dict:
    import rag_engine

    result = rag_engine.retrieve_legal_guidance_with_sources(question)
    return {
        "user_input": question,
        "response": result.answer,
        "retrieved_contexts": [result.context] if result.context else [],
        "sources": result.sources,
    }


def _split_chunks(context_text: str) -> list[str]:
    """Split a multi-chunk context string back into individual chunks."""
    if not context_text:
        return []
    parts = re.split(r'(?:^|\n)\[(\d+)\]\s+', context_text)
    chunks = []
    for i in range(1, len(parts), 2):
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            chunks.append(content)
    if not chunks and context_text.strip():
        chunks = [context_text.strip()]
    return chunks


def print_summary(results: dict[str, list[float | None]]):
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)

    for metric_name, scores in results.items():
        valid = [s for s in scores if s is not None]
        if valid:
            avg = sum(valid) / len(valid)
            min_s = min(valid)
            max_s = max(valid)
            print(f"\n  {metric_name:25s}  avg={avg:.3f}  min={min_s:.3f}  max={max_s:.3f}  (n={len(valid)})")
        else:
            print(f"\n  {metric_name:25s}  NO VALID SCORES")

    print("\n" + "-" * 60)

    all_valid = []
    for scores in results.values():
        all_valid.extend(s for s in scores if s is not None)
    if all_valid:
        overall = sum(all_valid) / len(all_valid)
        print(f"  OVERALL SCORE: {overall:.3f}")
    print("=" * 60)
    print()


def save_results(dataset: list[dict], results: dict[str, list], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["question", "response_preview", "reference_preview"]
        header += list(results.keys())
        writer.writerow(header)

        for i, sample in enumerate(dataset):
            row = [
                sample["user_input"],
                sample.get("response", "")[:100],
                sample.get("reference", "")[:100],
            ]
            for metric_name, scores in results.items():
                val = scores[i] if i < len(scores) else None
                row.append(f"{val:.3f}" if val is not None else "N/A")
            writer.writerow(row)

    print(f"  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for DGCA RAG system")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--samples", type=int, default=0, help="0 = all")
    parser.add_argument("--evaluator", choices=["auto", "azure", "ollama"], default="auto")
    parser.add_argument("--runs", type=int, default=VOTING_RUNS, help=f"Majority voting runs (default {VOTING_RUNS})")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"  Loaded {len(dataset)} samples")

    if args.samples > 0:
        dataset = dataset[: args.samples]
        print(f"  Evaluating first {args.samples} samples")

    print("\nRunning RAG queries...")
    rag_results = []
    for i, sample in enumerate(dataset):
        print(f"  [{i + 1}/{len(dataset)}] {sample['user_input'][:70]}...")
        rag_result = run_rag_query(sample["user_input"])
        merged = {**sample, **rag_result}
        rag_results.append(merged)

    print("\nBuilding evaluator LLM...")
    llm = build_llm_caller(args.evaluator)

    results: dict[str, list[float | None]] = {
        "faithfulness": [],
        "context_recall": [],
        "context_precision": [],
    }

    print(f"\nScoring with RAGAS metrics ({args.runs} voting runs per metric)...")
    total_start = time.time()

    for i, s in enumerate(rag_results):
        print(f"  [{i + 1}/{len(rag_results)}] {s['user_input'][:70]}...")
        sample_start = time.time()

        ctx_list = s.get("retrieved_contexts", [])
        context_text = ctx_list[0] if ctx_list else ""
        chunks = _split_chunks(context_text)

        f = score_with_voting(
            llm, metric_faithfulness, "faithfulness",
            s["user_input"], s["response"], s.get("reference", ""),
            chunks, args.runs,
        )
        cr = score_with_voting(
            llm, metric_context_recall, "context_recall",
            s["user_input"], s["response"], s.get("reference", ""),
            chunks, args.runs,
        )
        cp = score_with_voting(
            llm, metric_context_precision, "context_precision",
            s["user_input"], s["response"], s.get("reference", ""),
            chunks, args.runs,
        )

        results["faithfulness"].append(f)
        results["context_recall"].append(cr)
        results["context_precision"].append(cp)

        elapsed = time.time() - sample_start
        fv = f"{f:.3f}" if f is not None else "FAIL"
        crv = f"{cr:.3f}" if cr is not None else "FAIL"
        cpv = f"{cp:.3f}" if cp is not None else "FAIL"
        print(f"    F={fv}  CR={crv}  CP={cpv}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n  Total scoring time: {total_elapsed:.1f}s ({total_elapsed / len(rag_results):.1f}s per sample)")

    print_summary(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"ragas_results_{timestamp}.csv"
    save_results(rag_results, results, output_path)


if __name__ == "__main__":
    main()
