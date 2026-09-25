"""Đánh giá A/B dense-only và hybrid + RRF trên cùng golden dataset.

Chạy sau khi đã index corpus::

    python -m src.task4_chunking_indexing
    python -m src.evaluate

Script dùng Ragas 0.4 collections API và provider/model trong ``.env``. Kết
quả chi tiết được lưu thành JSON; báo cáo Markdown được sinh tự động.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .task10_generation import LLM_MODEL, SAFE_REFUSAL, answer_from_chunks
from .task4_chunking_indexing import EMBEDDING_MODEL
from .task5_semantic_search import semantic_search
from .task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve


load_dotenv()

ROOT = Path(__file__).parent.parent
EVALUATION_DIR = ROOT / "group_project" / "evaluation"
GOLDEN_PATH = EVALUATION_DIR / "golden_dataset.json"
RAW_RESULTS_PATH = EVALUATION_DIR / "raw_results.json"
REPORT_PATH = EVALUATION_DIR / "RESULT.md"
TOP_K = 5


def retrieve_config(question: str, config: str) -> list[dict]:
    if config == "dense-only":
        return semantic_search(question, top_k=TOP_K)
    return retrieve(question, top_k=TOP_K, use_reranking=True)


async def build_metrics():
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    provider = os.getenv("LLM_PROVIDER", "").lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", provider).lower()

    if provider == "gemini":
        from google import genai
        from ragas.embeddings import GoogleEmbeddings

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        judge = llm_factory(LLM_MODEL, provider="google", client=client)
        embeddings = GoogleEmbeddings(client=client, model=EMBEDDING_MODEL)
    elif provider == "openai":
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings

        # Ragas metric.ascore() cần async client cho cả judge lẫn embeddings.
        client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=5
        )
        judge = llm_factory(LLM_MODEL, provider="openai", client=client)
        if embedding_provider == "openai":
            embeddings = OpenAIEmbeddings(client=client, model=EMBEDDING_MODEL)
        else:
            from google import genai
            from ragas.embeddings import GoogleEmbeddings

            gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            embeddings = GoogleEmbeddings(client=gemini_client, model=EMBEDDING_MODEL)
    else:
        raise ValueError(
            f"Evaluator chưa hỗ trợ LLM_PROVIDER={provider!r}; dùng openai hoặc gemini"
        )

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevance": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_recall": ContextRecall(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
    }


async def score_case(metrics, case: dict, answer: str, contexts: list[str]) -> dict:
    common = {
        "user_input": case["question"],
        "response": answer,
        "retrieved_contexts": contexts,
    }
    calls = {
        "faithfulness": metrics["faithfulness"].ascore(**common),
        "answer_relevance": metrics["answer_relevance"].ascore(
            user_input=case["question"], response=answer
        ),
        "context_recall": metrics["context_recall"].ascore(
            user_input=case["question"],
            reference=case["expected_answer"],
            retrieved_contexts=contexts,
        ),
        "context_precision": metrics["context_precision"].ascore(
            user_input=case["question"],
            reference=case["expected_answer"],
            retrieved_contexts=contexts,
        ),
    }
    results = await asyncio.gather(*calls.values())
    return {
        name: float(result.value)
        for name, result in zip(calls, results, strict=True)
    }


async def score_case_with_retry(
    metrics, case: dict, answer: str, contexts: list[str], attempts: int = 3
) -> dict:
    """Chấm lại khi judge timeout; một lỗi mạng không được làm hỏng cả lượt chạy."""
    for attempt in range(1, attempts + 1):
        try:
            return await score_case(metrics, case, answer, contexts)
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"Scoring failed ({type(error).__name__}), retry {attempt}/{attempts - 1}")
            await asyncio.sleep(10 * attempt)
    return {}


async def evaluate() -> list[dict]:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    metrics = await build_metrics()
    rows: list[dict] = []
    for config in ("dense-only", "hybrid + RRF"):
        for index, case in enumerate(golden, 1):
            started = time.perf_counter()
            chunks = retrieve_config(case["question"], config)
            # Chấm trên thứ tự retrieval gốc: context precision phụ thuộc thứ hạng.
            ordered = chunks
            answer = answer_from_chunks(case["question"], chunks) or SAFE_REFUSAL
            # Latency chỉ gồm retrieval + generation, không tính thời gian judge chấm.
            latency = time.perf_counter() - started
            scores = await score_case_with_retry(
                metrics, case, answer, [item["content"] for item in ordered]
            )
            row = {
                "case": index,
                "config": config,
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "answer": answer,
                "source_ids": [item["id"] for item in ordered],
                "latency_seconds": round(latency, 3),
                **scores,
            }
            rows.append(row)
            RAW_RESULTS_PATH.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[{config}] {index}/{len(golden)}: {scores}")
    write_report(rows, len(golden))
    return rows


def _average(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_report(rows: list[dict], dataset_size: int) -> None:
    metric_fields = (
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
    )
    configs = {
        name: [row for row in rows if row["config"] == name]
        for name in ("dense-only", "hybrid + RRF")
    }
    averages = {
        config: {field: _average(items, field) for field in metric_fields}
        for config, items in configs.items()
    }
    overall = {
        config: statistics.fmean(scores.values())
        for config, scores in averages.items()
    }
    winner = max(overall, key=overall.get)
    worst = sorted(
        rows, key=lambda row: statistics.fmean(float(row[field]) for field in metric_fields)
    )[:3]
    weakest = min(
        metric_fields,
        key=lambda field: averages["hybrid + RRF"][field],
    )

    def metric_row(label: str, field: str) -> str:
        a = averages["dense-only"][field]
        b = averages["hybrid + RRF"][field]
        return f"| {label} | {a:.4f} | {b:.4f} | {b - a:+.4f} |"

    worst_rows = []
    for position, row in enumerate(worst, 1):
        scores = [row[field] for field in metric_fields]
        stage = "retrieval" if min(scores[2:]) <= min(scores[:2]) else "generation"
        worst_rows.append(
            f"| {position} | {row['question']} | {row['config']} | "
            f"{row['faithfulness']:.3f} | {row['answer_relevance']:.3f} | "
            f"{row['context_recall']:.3f} | {row['context_precision']:.3f} | "
            f"{stage} | Metric thấp nhất: {min(metric_fields, key=lambda f: row[f])} |"
        )

    latency = {config: _average(items, "latency_seconds") for config, items in configs.items()}
    report = f"""# RAG evaluation results — Du lịch Hà Nội

## Run information

| Field | Value |
|---|---|
| Evaluation date | {datetime.now(timezone.utc).isoformat()} |
| Framework and version | Ragas {version('ragas')} |
| Evaluator model | {LLM_MODEL} |
| Generator model | {LLM_MODEL} |
| Embedding model | {EMBEDDING_MODEL} |
| Corpus version/commit | {_git_commit()} |
| Golden dataset size | {dataset_size} |
| `top_k` | {TOP_K} |
| Fallback threshold and calibration | {SCORE_THRESHOLD} — best dense cosine của 15 câu golden: min 0.471 (Ba Vì–Suối Hai), max 0.737; 5 câu ngoài domain (nấu ăn, bitcoin, bóng đá, Python, thời tiết Paris): 0.254–0.421. Threshold đặt giữa hai nhóm. |

## Configurations

- **Config A — dense-only:** `{EMBEDDING_MODEL}` + cosine search, không BM25/RRF.
- **Config B — hybrid + RRF:** dense và BM25, fusion RRF một lần; PageIndex khi dense score dưới threshold.

Hai cấu hình dùng cùng golden dataset, generator, evaluator, prompt và `top_k`.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
{metric_row('Faithfulness', 'faithfulness')}
{metric_row('Answer relevance', 'answer_relevance')}
{metric_row('Context recall', 'context_recall')}
{metric_row('Context precision', 'context_precision')}
| **Average** | **{overall['dense-only']:.4f}** | **{overall['hybrid + RRF']:.4f}** | **{overall['hybrid + RRF'] - overall['dense-only']:+.4f}** |

## A/B comparison

- Cấu hình tốt hơn: **{winner}**.
- Evidence: điểm trung bình {overall[winner]:.4f} trên {dataset_size} câu grounded.
- Trade-off về latency/cost: dense-only {latency['dense-only']:.3f}s/câu; hybrid + RRF {latency['hybrid + RRF']:.3f}s/câu.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
{chr(10).join(worst_rows)}

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Tối ưu `{weakest}` bằng kiểm tra ba câu tệ nhất | Đây là metric thấp nhất của hybrid ({averages['hybrid + RRF'][weakest]:.4f}) | Tăng chất lượng tổng thể | Chạy lại đúng 15 câu và so delta |
| 2 | Hiệu chỉnh chunk size/overlap và threshold | Truy vấn địa danh và số liệu nằm ở nhiều loại tài liệu | Tăng recall, giảm fallback sai | Grid search trên golden dataset |
| 3 | Bổ sung nguồn giờ mở cửa/giá vé chính thức | Corpus hiện mạnh về lịch sử và quy hoạch hơn thông tin vận hành | Mở rộng phạm vi câu hỏi hữu ích | Thêm golden cases cho giờ/giá và chạy regression |

## Bonus experiments

Chưa chạy bonus. Chỉ triển khai HyDE, reranker nâng cao hoặc memory sau khi baseline ổn định.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(evaluate())
