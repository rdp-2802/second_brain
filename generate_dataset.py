"""
Generates synthetic training data for the retrieved-context relevance classifier
using the Gemini API. Designed to run locally (this is I/O-bound, not GPU-bound).

Usage:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python generate_dataset.py

Output:
    Appends one JSON object per line to dataset_groups.jsonl as groups complete,
    so progress is never lost even if the script is interrupted or rate-limited
    partway through. Safe to stop and re-run (it resumes from existing lines).
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv


from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration — adjust these once you've confirmed current Gemini free-tier
# limits for your chosen model. Defaults below are deliberately conservative.
# ---------------------------------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Pro models were removed from the Gemini free tier in 2026 — free tier is
# Flash-only now. Per your account's live quota dashboard, general Flash
# models (3.5/3.6/3.7/3.8 Flash, 2.5 Flash) are capped at 20 requests/day —
# unworkable for bulk generation. Flash-Lite models (3.1 and 3.5) show
# 15 RPM / 500 RPD, which comfortably covers 300 groups in one run.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

OUTPUT_FILE = Path("dataset_groups.jsonl")
TOTAL_GROUPS_TARGET = 300

# Confirmed limits for gemini-3.1-flash-lite on your account: 15 RPM, 500 RPD.
# Leave margin on both so retries and any account variance don't tip you over.
REQUESTS_PER_MINUTE_CAP = 12
MAX_CONCURRENT_REQUESTS = 3
DAILY_REQUEST_BUDGET = 450  # stop dispatching new calls once this many attempts are used today

MAX_RETRIES = 5

DOMAINS = [
    "Career (job search, workplace situations, ambitions, skill-building)",
    "Relationships and breakups (dating, conflict, moving on)",
    "Insecurities and self-doubt (body image, comparison, confidence)",
    "Goals and personal growth (habits, discipline, long-term plans)",
    "Physique and fitness (training, diet, body changes over time)",
    "Books read and ideas absorbed from them (reflections, applying lessons)",
    "Sports (personal play, following a team, fitness-adjacent competition)",
]

PROMPT_TEMPLATE = """You are generating synthetic training data for a text classifier used in a personal
knowledge-base RAG system. Given a conversation state, the classifier judges whether
each of several retrieved pieces of personal knowledge is useful for answering the
user's latest query.

### Structure of one "group"
Each group has ONE shared conversation state, and TEN candidate retrieved-context
pieces evaluated against that same shared state:

- conversation_summary — a summary of everything discussed in this chat before the recent messages. Make this realistically long: 7-8 paragraphs, as if a long-running personal conversation has been condensed.
- recent_messages — the last 15-20 chat turns (alternating user/assistant), not yet summarized, following on naturally from the summary.
- query — the user's single latest message.
- candidates — an array of exactly 10 objects, each representing one retrieved piece of personal knowledge evaluated against the SAME summary/messages/query above. Each candidate has:
  - retrieved_context — 1-3 sentences of a specific fact, preference, or past note from the user's personal knowledge base
  - label — one of "needed", "redundant", "irrelevant" (see definitions below)
  - relative_rank — an integer 1-10 giving this candidate's usefulness rank within this group (1 = most useful, 10 = least useful). Ties are not allowed; all 10 ranks must be used exactly once per group.

### Label definitions
- needed — this retrieved_context contains information required to answer the query well, and that information is NOT already present in conversation_summary or recent_messages.
- redundant — this retrieved_context contains information already present (even if reworded) in conversation_summary or recent_messages. It may be topically related to the query, but adds nothing new.
- irrelevant — this retrieved_context is about a different topic than the query and conversation, and would not help answer it at all.

### Construction rules for the 10 candidates within a group
- Include roughly 3-4 needed, 3 redundant, and 3-4 irrelevant candidates per group (exact split can vary slightly group to group).
- Among the needed candidates, vary how directly useful they are (assign relative_rank accordingly).
- Among the redundant candidates, vary how closely they paraphrase what's already in summary/recent_messages.
- Among the irrelevant candidates, vary how far off-topic they are.
- The whole group should center on a single realistic personal-life scenario within the domain focus below.

### Domain focus for this batch
Center this entire group's scenario on: {{DOMAIN_FOCUS}}

### Output format
Return ONLY a raw JSON object. No markdown code fences, no explanation, no preamble.
Follow this exact shape:

{
  "conversation_summary": "string, 7-8 paragraphs",
  "recent_messages": ["string", "string ... 15-20 total"],
  "query": "string",
  "candidates": [
    {"retrieved_context": "string", "label": "needed", "relative_rank": 1}
  ]
}

The candidates array must have exactly 10 objects with ranks 1-10 each used once.
Generate one full group now, following all rules above.
"""

client = genai.Client(api_key=GEMINI_API_KEY)
generation_config = types.GenerateContentConfig(
    temperature=1.0,
    max_output_tokens=8192,
)


class RateLimiter:
    """Caps requests to REQUESTS_PER_MINUTE_CAP by spacing out call start times."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self._last_call = time.monotonic()


rate_limiter = RateLimiter(REQUESTS_PER_MINUTE_CAP)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
write_lock = asyncio.Lock()

daily_calls_lock = asyncio.Lock()
daily_calls_made = 0
daily_budget_exhausted = False


async def try_reserve_daily_call() -> bool:
    """Returns True and reserves one call against today's budget, or False if
    the daily budget is already used up (in which case no API call is made)."""
    global daily_calls_made, daily_budget_exhausted
    async with daily_calls_lock:
        if daily_calls_made >= DAILY_REQUEST_BUDGET:
            daily_budget_exhausted = True
            return False
        daily_calls_made += 1
        return True


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def validate_group(obj: dict) -> bool:
    required_keys = {"conversation_summary", "recent_messages", "query", "candidates"}
    if not required_keys.issubset(obj.keys()):
        return False
    candidates = obj["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 10:
        return False
    try:
        ranks = sorted(c["relative_rank"] for c in candidates)
    except (KeyError, TypeError):
        return False
    if ranks != list(range(1, 11)):
        return False
    labels = {c.get("label") for c in candidates}
    if not labels.issubset({"needed", "redundant", "irrelevant"}):
        return False
    return True


async def generate_one_group(domain: str, index: int) -> dict | None:
    prompt = PROMPT_TEMPLATE.replace("{{DOMAIN_FOCUS}}", domain)

    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            if not await try_reserve_daily_call():
                print(f"[{index}] daily request budget reached ({DAILY_REQUEST_BUDGET}), stopping this group — resume tomorrow")
                return None
            await rate_limiter.wait()
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL_NAME,
                    contents=prompt,
                    config=generation_config,
                )
                raw_text = strip_code_fences(response.text)
                obj = json.loads(raw_text)
                if validate_group(obj):
                    obj["_domain"] = domain
                    obj["_index"] = index
                    return obj
                print(f"[{index}] validation failed (attempt {attempt}), retrying")
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "quota" in err.lower() or "rate" in err.lower()
                backoff = (2 ** attempt) + random.uniform(0, 1)
                kind = "rate limited" if is_rate_limit else "error"
                print(f"[{index}] {kind}: {err[:200]} — backing off {backoff:.1f}s (attempt {attempt})")
                await asyncio.sleep(backoff)

    print(f"[{index}] FAILED after {MAX_RETRIES} attempts, skipping")
    return None


async def append_result(obj: dict):
    async with write_lock:
        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def worker(domain: str, index: int):
    result = await generate_one_group(domain, index)
    if result:
        await append_result(result)


async def main():
    already_done = 0
    if OUTPUT_FILE.exists():
        already_done = sum(1 for _ in OUTPUT_FILE.open(encoding="utf-8"))
        print(f"Found {already_done} existing groups in {OUTPUT_FILE}, continuing from there.")

    remaining = TOTAL_GROUPS_TARGET - already_done
    if remaining <= 0:
        print("Target already reached.")
        return

    tasks = [
        worker(DOMAINS[i % len(DOMAINS)], already_done + i)
        for i in range(remaining)
    ]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 10 == 0:
            print(f"progress: {completed}/{remaining} new groups this run")
        if daily_budget_exhausted:
            print("Daily request budget reached — stopping early. Re-run tomorrow to continue from where you left off.")
            break

    print(f"Done. Total groups in {OUTPUT_FILE}: {already_done + completed}")


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        raise SystemExit("Set GEMINI_API_KEY environment variable before running.")
    asyncio.run(main())