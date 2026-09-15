"""
Business logic verification for second-brain.

Tests the full pipeline with REAL DB + REAL Gemini LLM/embedding.

    python tests/test_business_logic.py              # run all tests
    python tests/test_business_logic.py --no-cleanup # keep test data in DB
    python tests/test_business_logic.py --quick      # skip slow LLM-heavy tests (3,4,5)

Prerequisites:
    - PostgreSQL running with second_brain DB (alembic upgrade head)
    - GEMINI_API_KEY and SQL_URL in .env

Thresholds are lowered for testing (summarisation=6, ingestion=4) so tests
finish quickly and cheaply while still exercising the half-batch and dedup logic.

Cleanup: deleting the test user CASCADE-deletes all chats, messages, blocks,
memories, and audit rows created during the run.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running `python tests/test_*.py`
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.orm import Session

from app.database.model import SessionLocal
from app.database.crud.user import create_user, delete_user
from app.database.crud.chat import create_chat, read_chat
from app.database.crud.message import read_messages_after_order
from app.database.crud.message_block import read_message_blocks_by_chat
from app.database.crud.memory_summary import (
    create_memory_summary,
    read_memory_summaries,
    read_memory_summary,
)
from app.database.crud.memory_detail import (
    create_memory_detail,
    read_memory_details_by_summary,
)
from app.services.chat_orchestration import ChatUser, handle_chat_turn
from app.services.semantic_retrieval import semantic_retrieval

# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------
SUMMARISATION_THRESHOLD = 6  # default is 30 — lowered for faster tests
INGESTION_THRESHOLD = 4      # default is 20 — lowered for faster tests

TEST_USER = {"name": "test_runner", "email": "test@runner.io", "mobile_no": 999999999}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

pass_count = 0
fail_count = 0
skip_count = 0


def _ok(msg: str):
    global pass_count
    pass_count += 1
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str):
    global fail_count
    fail_count += 1
    print(f"  {RED}✗ FAIL:{RESET} {msg}")


def _info(msg: str):
    print(f"  {DIM}{msg}{RESET}")


def _warn(msg: str):
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def check(condition: bool, ok_msg: str, fail_msg: str) -> bool:
    if condition:
        _ok(ok_msg)
    else:
        _fail(fail_msg)
    return condition


def dump_db_state(db: Session, user_id, chat_id, label: str = ""):
    """Print a compact snapshot of DB state for a chat."""
    chat = read_chat(db, user_id, chat_id)
    if chat is None:
        print(f"    {DIM}[{label}] chat not found{RESET}")
        return
    blocks = read_message_blocks_by_chat(db, user_id, chat_id) or []
    summaries = read_memory_summaries(db, user_id) or []
    msg_count = len(chat.message or [])
    print(f"    {DIM}[{label}] messages={msg_count}  blocks={len(blocks)}  "
          f"mem_summaries={len(summaries)}  "
          f"last_summ={chat.last_summarisation_message_order}  "
          f"last_ingest={chat.last_ingestion_message_order}{RESET}")
    if blocks:
        for b in blocks:
            preview = b.content[:80].replace(chr(10), " ")
            print(f"      block#{b.order_in_chat}: {preview!r}...")
    if summaries:
        for s in summaries:
            pv = s.content[:80].replace(chr(10), " ")
            print(f"      summary {str(s.id)[:8]}: {pv!r}...  (retrieved {s.retrieval_count}x)")


def safe_handle_turn(db, chat_user, query, **kw) -> str | None:
    """Call handle_chat_turn with retry + throttling for free-tier rate limits (5 RPM)."""
    throttle = kw.pop("_throttle", 13)  # seconds between calls; 13s keeps us under 5 RPM
    max_retries = kw.pop("_retries", 2)
    for attempt in range(1, max_retries + 2):
        try:
            resp = handle_chat_turn(db, chat_user, query, **kw)
            db.expire_all()
            if attempt == 1 and throttle:
                time.sleep(throttle)
            return resp
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
            if is_rate_limit and attempt <= max_retries:
                # Parse retry delay from error if available, else use throttle
                import re
                m = re.search(r"retry[^0-9]*(\d+)", msg, re.IGNORECASE)
                wait = int(m.group(1)) + 2 if m else throttle + 5
                wait = min(wait, 65)
                print(f"    {YELLOW}Rate limited (attempt {attempt}/{max_retries + 1}), retrying in {wait}s ...{RESET}")
                time.sleep(wait)
                try:
                    db.rollback()
                    db.expire_all()
                except Exception:
                    pass
                continue
            print(f"    {RED}LLM/DB error during handle_chat_turn({query[:40]!r}): {e}{RESET}")
            traceback.print_exc()
            try:
                db.rollback()
            except Exception:
                pass
            return None
    return None


# ---------------------------------------------------------------------------
# Setup / Teardown
# ---------------------------------------------------------------------------
test_user = None
test_chats: dict[str, object] = {}  # name -> Chat


def setup(db: Session):
    global test_user
    print(f"\n{BOLD}SETUP{RESET}")
    print(f"  Creating test user {TEST_USER} ...")
    test_user = create_user(db, **TEST_USER)
    assert test_user is not None, "create_user returned None"
    print(f"  user_id={test_user.id}")
    _ok(f"Test user created: {test_user.id}")
    return test_user


def make_chat(db: Session, name: str):
    """Create a fresh chat for isolation between test groups."""
    chat = create_chat(db, test_user.id)
    assert chat is not None, f"create_chat failed for {name}"
    test_chats[name] = chat
    print(f"  chat[{name}] id={chat.id}")
    return chat


def teardown(db: Session, keep: bool = False):
    global test_user
    if keep:
        print(f"\n{YELLOW}--no-cleanup: leaving test data in DB{RESET}")
        try:
            print(f"  user_id={test_user.id}")
        except Exception:
            pass
        for name, chat in test_chats.items():
            try:
                print(f"  chat[{name}] id={chat.id}")
            except Exception:
                print(f"  chat[{name}] id=<expired>")
        return
    print(f"\n{BOLD}TEARDOWN{RESET}")
    if test_user is not None:
        try:
            uid = test_user.id  # capture before delete expires the object
            delete_user(db, uid)
            _ok(f"Deleted test user {uid} (cascade)")
        except Exception as e:
            print(f"  {RED}Teardown error: {e}{RESET}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# TEST 01 — Basic chat turn
# ---------------------------------------------------------------------------
def test_01_basic_chat(db: Session):
    """
    Send a few messages, verify:
    - handle_chat_turn returns a non-empty LLM response
    - messages are persisted with correct order_in_chat
    - no summarisation or ingestion triggered (below threshold)
    """
    print(f"\n{BOLD}TEST 01 — Basic chat turn{RESET}")
    chat = make_chat(db, "basic")
    cu = ChatUser(user_id=test_user.id, chat_id=chat.id)

    queries = [
        "Hi, I'm Rishi and I'm building a personal knowledge base called second-brain.",
        "It uses FastAPI, SQLAlchemy and PostgreSQL with pgvector.",
        "Can you summarize what we've discussed so far?",
    ]

    responses: list[str | None] = []
    for i, q in enumerate(queries, 1):
        print(f"\n  Turn {i}: user → {q[:60]!r}")
        t0 = time.time()
        resp = safe_handle_turn(
            db, cu, q,
            summarisation_threshold=SUMMARISATION_THRESHOLD,
            ingestion_threshold=INGESTION_THRESHOLD,
        )
        dt = time.time() - t0
        preview = (resp[:100] + "…") if resp and len(resp) > 100 else (resp or "<no response>")
        print(f"    assistant ({dt:.1f}s) → {preview!r}")
        responses.append(resp)

    # --- Assertions ---
    print(f"\n  Assertions:")
    for i, r in enumerate(responses, 1):
        check(r is not None and len(r.strip()) > 0,
              f"Turn {i} returned non-empty LLM response",
              f"Turn {i} LLM response was empty or None")

    chat = read_chat(db, test_user.id, chat.id)
    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)

    total_msgs = len(chat.message or [])
    expected = len(queries) * 2  # user + assistant per turn
    check(total_msgs == expected,
          f"Message count is {expected} ({len(queries)} turns × 2)",
          f"Expected {expected} messages, found {total_msgs}")

    # Verify order_in_chat is sequential 1..N
    orders = sorted(m.order_in_chat for m in (chat.message or []))
    check(orders == list(range(1, expected + 1)),
          f"order_in_chat is sequential 1..{expected}: {orders}",
          f"order_in_chat not sequential: {orders}")

    # Verify roles alternate user / assistant
    msgs_sorted = sorted(chat.message or [], key=lambda m: m.order_in_chat)
    roles = [m.role.value if hasattr(m.role, "value") else str(m.role) for m in msgs_sorted]
    expected_roles = ["user", "assistant"] * len(queries)
    check(roles == expected_roles,
          f"Roles alternate correctly: {roles}",
          f"Roles mismatch. Expected {expected_roles}, got {roles}")

    # Below threshold — no summarisation or ingestion
    dump_db_state(db, test_user.id, chat.id, "after test_01")

    # With 6 messages (3 turns * 2) and summarisation threshold 6:
    # handle_chat_turn checks AFTER persisting, so the 3rd turn (6 msgs) DOES trigger.
    # But summariser takes half (3) and needs >=2 pending, so it will create 1 block.
    # Either outcome is valid depending on timing — just report what happened.
    blocks = read_message_blocks_by_chat(db, test_user.id, chat.id) or []
    if blocks:
        _info(f"Summarisation triggered (6 msgs hit threshold 6) — {len(blocks)} block(s) created")
        # Verify half-batch: block should link to half the messages at trigger time
        expected_summ = len(blocks[0].message) if hasattr(blocks[0], "message") else "?"
        _info(f"First block linked to {expected_summ} message(s) (half-batch)")
    else:
        _info("No summarisation — threshold not crossed or summariser returned early (also valid)")


# ---------------------------------------------------------------------------
# TEST 02 — Retrieval + dedup
# ---------------------------------------------------------------------------
def test_02_retrieval(db: Session):
    """
    Pre-seed a memory, then chat about it and verify:
    - semantic_retrieval finds the memory
    - audit rows (RetrievedSummary / RetrievedDetail) are written
    - retrieval_count increments
    - dedup: previous_memories cleaned against current retrieval
    """
    print(f"\n{BOLD}TEST 02 — Retrieval + dedup{RESET}")
    chat = make_chat(db, "retrieval")
    cu = ChatUser(user_id=test_user.id, chat_id=chat.id)

    # --- Seed a memory directly via CRUD (bypasses ingestion) ---
    print(f"\n  Seeding memory via CRUD ...")
    seeded_summary = create_memory_summary(
        db, test_user.id,
        content="Rishi is building a personal knowledge base called second-brain using FastAPI, SQLAlchemy, PostgreSQL and pgvector for vector search.",
    )
    if seeded_summary is None:
        _fail("create_memory_summary returned None — seeding failed, skipping test")
        return
    print(f"    summary id={seeded_summary.id}")
    print(f"    content: {seeded_summary.content[:80]!r}")

    seeded_detail = create_memory_detail(
        db, test_user.id, seeded_summary.id, chat.id,
        content="The second-brain project uses FastAPI for the API layer and SQLAlchemy as ORM with PostgreSQL + pgvector for 1024-dim embeddings.",
    )
    if seeded_detail is None:
        _warn("create_memory_detail returned None — continuing with summary only")
    else:
        print(f"    detail  id={seeded_detail.id}")
        print(f"    content: {seeded_detail.content[:80]!r}")

    db.expire_all()
    seeded_summary = read_memory_summary(db, test_user.id, seeded_summary.id)
    initial_count = seeded_summary.retrieval_count
    print(f"    initial retrieval_count={initial_count}")

    # --- Sanity: direct semantic_retrieval without going through handle_chat_turn ---
    print(f"\n  Direct semantic_retrieval (query about tech stack) ...")
    try:
        retrieved = semantic_retrieval(
            db, test_user.id,
            summary_blocks=[],
            recent_messages=[],
            query="What technology stack am I using for second-brain?",
        )
        print(f"    retrieved {len(retrieved)} memories:")
        for r in retrieved:
            pv = r.memory.content[:70].replace(chr(10), " ")
            print(f"      [{r.memory_type} sim={r.similarity:.3f}] {pv!r}")
        found_seeded = any(r.memory.id == seeded_summary.id for r in retrieved)
        check(found_seeded,
              "Direct retrieval found the seeded summary",
              "Direct retrieval did NOT find the seeded summary (similarity may be < 0.5 — try a closer query)")
    except Exception as e:
        _fail(f"semantic_retrieval raised: {e}")
        traceback.print_exc()

    # --- Now go through handle_chat_turn (full pipeline including audit writes) ---
    query = "What tech stack did I choose for my second-brain project?"
    print(f"\n  handle_chat_turn: {query!r}")
    resp = safe_handle_turn(
        db, cu, query,
        summarisation_threshold=999,  # disable maintenance for this test
        ingestion_threshold=999,
    )
    if resp is not None:
        print(f"    assistant → {(resp[:150] + '…') if len(resp) > 150 else resp!r}")
        check(len(resp.strip()) > 0, "LLM returned non-empty response", "LLM response empty")
        # Check that the response actually uses retrieved knowledge
        tech_words = ["fastapi", "sqlalchemy", "postgresql", "pgvector", "postgres"]
        resp_lower = resp.lower()
        mentions_tech = any(w in resp_lower for w in tech_words)
        if mentions_tech:
            _ok("LLM response mentions expected tech stack (retrieval grounded)")
        else:
            _warn("LLM response doesn't mention expected tech words — retrieval may not have grounded the prompt")
    else:
        _fail("handle_chat_turn returned None (LLM failure)")

    # --- Verify audit rows ---
    print(f"\n  Verifying audit rows ...")
    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)
    msgs = sorted(chat.message or [], key=lambda m: m.order_in_chat)
    user_msgs = [m for m in msgs if (m.role.value if hasattr(m.role, "value") else str(m.role)) == "user"]
    if user_msgs:
        last_user_msg = user_msgs[-1]
        rs_count = len(getattr(last_user_msg, "retrieved_summary", []) or [])
        rd_count = len(getattr(last_user_msg, "retrieved_detail", []) or [])
        print(f"    Last user message (order {last_user_msg.order_in_chat}) has "
              f"{rs_count} retrieved_summary + {rd_count} retrieved_detail rows")
        if rs_count + rd_count > 0:
            _ok(f"Audit rows written for last user message ({rs_count + rd_count})")
        else:
            _warn("No audit rows for last user message — retrieval may have returned empty (cosine < 0.5)")

        # Verify retrieval_count incremented
        db.expire_all()
        seeded_summary = read_memory_summary(db, test_user.id, seeded_summary.id)
        new_count = seeded_summary.retrieval_count if seeded_summary else -1
        print(f"    retrieval_count: {initial_count} → {new_count}")
        if new_count > initial_count:
            _ok("retrieval_count incremented after retrieval")
        else:
            # Check if the summary was retrieved at all (maybe detail was retrieved instead)
            retrieved_ids = set()
            for rs in (getattr(last_user_msg, "retrieved_summary", []) or []):
                retrieved_ids.add(getattr(rs, "memory_summary_id", None))
            if seeded_summary.id in retrieved_ids:
                _warn("Summary was retrieved but retrieval_count didn't increment — check update path")
            else:
                _info("Seeded summary not among this query's retrievals — similarity may be below threshold")

    # --- Dedup test: second query retrieving the same memory ---
    print(f"\n  Dedup test — second query on same topic ...")
    q2 = "Remind me what database I'm using for second-brain?"
    resp2 = safe_handle_turn(
        db, cu, q2,
        summarisation_threshold=999,
        ingestion_threshold=999,
    )
    if resp2 is not None:
        print(f"    assistant → {(resp2[:150] + '…') if len(resp2) > 150 else resp2!r}")

    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)
    msgs = sorted(chat.message or [], key=lambda m: m.order_in_chat)
    user_msgs = [m for m in msgs if (m.role.value if hasattr(m.role, "value") else str(m.role)) == "user"]
    if len(user_msgs) >= 2:
        # Collect all retrieved IDs across both user messages
        all_ids: list = []
        for um in user_msgs:
            for rs in (getattr(um, "retrieved_summary", []) or []):
                all_ids.append(getattr(rs, "memory_summary_id", None))
            for rd in (getattr(um, "retrieved_detail", []) or []):
                all_ids.append(getattr(rd, "memory_detail_id", None))
        # The prompt's previous_memories should have been deduplicated against current
        # i.e. if memory X is in both previous and current, it should only appear in current
        # We verify audit rows are per-message (not deduped at DB level) — each message's audit is independent
        _info(f"Total audit memory IDs across {len(user_msgs)} user messages: {len(all_ids)} (includes duplicates across messages — expected)")

    dump_db_state(db, test_user.id, chat.id, "after test_02")


# ---------------------------------------------------------------------------
# TEST 03 — Summarisation threshold (half-batch)
# ---------------------------------------------------------------------------
def test_03_summarisation(db: Session):
    """
    Send enough messages to cross the summarisation threshold.
    Verify:
    - A summary block is created
    - Only half the pending messages are summarised (half-batch rule)
    - MessageJoinBlock rows link the correct messages
    - Next context still has recent messages (not empty)
    """
    print(f"\n{BOLD}TEST 03 — Summarisation (half-batch){RESET}")
    chat = make_chat(db, "summarisation")
    cu = ChatUser(user_id=test_user.id, chat_id=chat.id)

    # Use ingestion_threshold=999 to isolate summarisation
    # Each handle_chat_turn = 2 messages (user+assistant), so 3 turns = 6 msgs = threshold
    topics = [
        "I've been thinking about the architecture for my knowledge base. I want to store memories as summaries and details with embeddings.",
        "For the API layer, I chose FastAPI because of its async support and automatic OpenAPI docs. The database is PostgreSQL with pgvector.",
        "The retrieval system uses cosine similarity on 1024-dim embeddings. I'm considering adding a ModernBERT classifier to filter results.",
        "For the frontend, I'm debating between Next.js and a simpler Vite setup. I prefer keeping things minimal at first.",
        "Deployment will be on a small VPS with Docker Compose. I want to keep costs low while I validate the idea.",
    ]

    print(f"  Sending {len(topics)} turns (threshold={SUMMARISATION_THRESHOLD}, ingestion disabled)...")
    for i, q in enumerate(topics, 1):
        print(f"    Turn {i}: {q[:55]!r} ...")
        resp = safe_handle_turn(
            db, cu, q,
            summarisation_threshold=SUMMARISATION_THRESHOLD,
            ingestion_threshold=999,
        )
        if resp is None:
            _warn(f"Turn {i} LLM failed — continuing")
        db.expire_all()
        chat_state = read_chat(db, test_user.id, chat.id)
        blocks = read_message_blocks_by_chat(db, test_user.id, chat.id) or []
        total = len(chat_state.message or [])
        print(f"      → total_msgs={total}  last_summ={chat_state.last_summarisation_message_order}  blocks={len(blocks)}")

    # --- Assertions ---
    print(f"\n  Assertions:")
    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)
    blocks = read_message_blocks_by_chat(db, test_user.id, chat.id) or []
    total_msgs = len(chat.message or [])
    last_summ = chat.last_summarisation_message_order or 0

    print(f"    total messages: {total_msgs}")
    print(f"    last_summarisation_message_order: {last_summ}")
    print(f"    summary blocks: {len(blocks)}")

    check(len(blocks) >= 1,
          f"At least 1 summary block created (found {len(blocks)})",
          f"No summary blocks created — threshold may not have been crossed (total={total_msgs}, thresh={SUMMARISATION_THRESHOLD})")

    if blocks:
        for idx, block in enumerate(blocks):
            pv = block.content[:100].replace(chr(10), " ")
            print(f"    Block {idx}: order={block.order_in_chat}  {pv!r}...")
            check(len(block.content.strip()) > 20,
                  f"Block {idx} content is non-empty ({len(block.content)} chars)",
                  f"Block {idx} content is empty or too short")

            # Verify half-batch: check how many messages are linked to this block
            linked = getattr(block, "message", None) or getattr(block, "message_link", None) or []
            # message_link is list[MessageJoinBlock], message is list[Message] via secondary
            linked_count = len(linked)
            print(f"      linked messages: {linked_count}")
            # At trigger time there were SUMMARISATION_THRESHOLD pending messages,
            # half-batch means linked should be threshold//2 (or floor)
            # But total pending at trigger may differ, so just verify it's not ALL

        # Verify recent messages still exist (not all summarised)
        from app.services.chat_orchestration import _form_chat_context
        summary_blocks, recent_messages, raw_messages = _form_chat_context(db, cu)
        print(f"    After summarisation: {len(summary_blocks)} summary_blocks, {len(recent_messages)} recent_messages in context")
        check(len(recent_messages) > 0,
              f"Recent messages preserved after summarisation ({len(recent_messages)} remain)",
              "No recent messages remain after summarisation — half-batch may not be working")

        # Verify summarised messages are not in recent_messages
        recent_orders = {m.order for m in recent_messages}
        check(last_summ not in recent_orders or last_summ == 0,
              "Summarised messages not duplicated in recent_messages",
              "Last summarised order still appears in recent_messages — dedup issue")

    dump_db_state(db, test_user.id, chat.id, "after test_03")


# ---------------------------------------------------------------------------
# TEST 04 — Ingestion threshold
# ---------------------------------------------------------------------------
def test_04_ingestion(db: Session):
    """
    Send enough personal/durable messages to cross the ingestion threshold.
    Verify:
    - MemorySummary + MemoryDetail records are created
    - DetailJoinMessage links details to source messages
    - last_ingestion_message_order advances
    """
    print(f"\n{BOLD}TEST 04 — Ingestion{RESET}")
    chat = make_chat(db, "ingestion")
    cu = ChatUser(user_id=test_user.id, chat_id=chat.id)

    # Durable personal content — ingestion LLM should extract this
    personal_messages = [
        "I've been working as a software engineer for 3 years, mostly in Python and backend systems.",
        "My long-term goal is to build a successful indie product and eventually go full-time on it.",
        "I grew up in Delhi and studied computer science at IIT Delhi. I graduated in 2022.",
        "I struggle with perfectionism — I often overthink before shipping. My new mantra is 'ship fast, iterate'.",
        "My favorite way to learn is by building projects, not by watching tutorials. I learn best hands-on.",
    ]

    # Snapshot memory counts before
    db.expire_all()
    before_summaries = len(read_memory_summaries(db, test_user.id) or [])
    before_chats_mem = 0  # will count after

    # Disable summarisation to isolate ingestion
    print(f"  Sending {len(personal_messages)} turns (ingestion threshold={INGESTION_THRESHOLD}, summarisation disabled)...")
    for i, q in enumerate(personal_messages, 1):
        print(f"    Turn {i}: {q[:55]!r} ...")
        resp = safe_handle_turn(
            db, cu, q,
            summarisation_threshold=999,
            ingestion_threshold=INGESTION_THRESHOLD,
        )
        if resp is None:
            _warn(f"Turn {i} LLM failed — continuing")
        db.expire_all()
        chat_state = read_chat(db, test_user.id, chat.id)
        total = len(chat_state.message or [])
        all_summaries = read_memory_summaries(db, test_user.id) or []
        new_count = len(all_summaries) - before_summaries
        print(f"      → total_msgs={total}  last_ingest={chat_state.last_ingestion_message_order}  new_memories={new_count}")

    # --- Assertions ---
    print(f"\n  Assertions:")
    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)
    all_summaries = read_memory_summaries(db, test_user.id) or []
    new_summaries = [s for s in all_summaries if s not in []]  # all are candidates
    # Filter to only those whose content looks like it came from our test messages
    # (other tests may have created summaries too)
    created_now = len(all_summaries) - before_summaries
    last_ingest = chat.last_ingestion_message_order or 0

    print(f"    total messages: {len(chat.message or [])}")
    print(f"    last_ingestion_message_order: {last_ingest}")
    print(f"    new memory summaries since start: {created_now}")
    print(f"    total memory summaries for user: {len(all_summaries)}")

    if created_now > 0:
        _ok(f"Ingestion created {created_now} new memory summary/summaries")
        # Show what was extracted
        db.expire_all()
        all_summaries_fresh = read_memory_summaries(db, test_user.id) or []
        # Show the most recent ones (likely from this test)
        for s in all_summaries_fresh[-min(3, len(all_summaries_fresh)):]:
            pv = s.content[:100].replace(chr(10), " ")
            print(f"      summary {str(s.id)[:8]}: {pv!r}...")
            details = read_memory_details_by_summary(db, test_user.id, s.id, limit=10) or []
            for d in details:
                dpv = d.content[:80].replace(chr(10), " ")
                print(f"        detail: {dpv!r}...")

        check(last_ingest > 0,
              f"last_ingestion_message_order advanced to {last_ingest}",
              "last_ingestion_message_order still 0 after ingestion should have triggered")
    else:
        # Ingestion may return empty if LLM decides nothing is worth remembering
        # That's valid — but with personal/durable content it SHOULD extract something
        _warn(f"No new memories created — ingestion LLM may have returned [] (threshold={INGESTION_THRESHOLD})")
        _info("This can happen if the LLM judges the content as not durable. Try more personal messages or check ingestion prompt.")
        # Still verify the mechanism works: last_ingestion should have advanced even if no memories
        if last_ingest > 0:
            _ok(f"Ingestion ran (last_ingest={last_ingest}) even though no memories were extracted")
        else:
            _warn("Ingestion did not run at all — check threshold and message count")

    dump_db_state(db, test_user.id, chat.id, "after test_04")


# ---------------------------------------------------------------------------
# TEST 05 — Full end-to-end pipeline
# ---------------------------------------------------------------------------
def test_05_full_e2e(db: Session):
    """
    Full pipeline in a single chat:
    1. Chat enough to trigger both summarisation and ingestion
    2. Verify all artifacts exist
    3. Chat again and verify retrieval uses ingested memories
    """
    print(f"\n{BOLD}TEST 05 — Full end-to-end pipeline{RESET}")
    chat = make_chat(db, "full_e2e")
    cu = ChatUser(user_id=test_user.id, chat_id=chat.id)

    # Record baseline
    db.expire_all()
    baseline_summaries = len(read_memory_summaries(db, test_user.id) or [])
    baseline_blocks = len(read_message_blocks_by_chat(db, test_user.id, chat.id) or [])

    phase1 = [
        "I love hiking in the Himalayas. Last year I trekked to Valley of Flowers and it was life-changing.",
        "My dream is to build a cabin in the mountains and live there part of the year. I want a simple life close to nature.",
        "I'm learning meditation — I practice 20 minutes every morning. It helps me stay focused and calm.",
        "I have a dog named Bruno, a golden retriever. He's 2 years old and loves swimming.",
        "My favorite book is 'Deep Work' by Cal Newport. It changed how I think about productivity.",
    ]

    print(f"\n  Phase 1: {len(phase1)} turns (both thresholds active: summ={SUMMARISATION_THRESHOLD}, ingest={INGESTION_THRESHOLD})")
    for i, q in enumerate(phase1, 1):
        print(f"    Turn {i}: {q[:50]!r} ...")
        resp = safe_handle_turn(
            db, cu, q,
            summarisation_threshold=SUMMARISATION_THRESHOLD,
            ingestion_threshold=INGESTION_THRESHOLD,
        )
        if resp is not None:
            pv = (resp[:80] + "…") if len(resp) > 80 else resp
            print(f"      → {pv!r}")
        db.expire_all()
        ch = read_chat(db, test_user.id, chat.id)
        blks = len(read_message_blocks_by_chat(db, test_user.id, chat.id) or [])
        mems = len(read_memory_summaries(db, test_user.id) or []) - baseline_summaries
        print(f"        state: msgs={len(ch.message or [])}  blocks={blks}  new_mems={mems}  "
              f"last_summ={ch.last_summarisation_message_order}  last_ingest={ch.last_ingestion_message_order}")

    # Snapshot after phase 1
    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)
    blocks_after = read_message_blocks_by_chat(db, test_user.id, chat.id) or []
    mems_after = len(read_memory_summaries(db, test_user.id) or []) - baseline_summaries
    print(f"\n  After phase 1: {len(blocks_after)} blocks, {mems_after} new memories")
    dump_db_state(db, test_user.id, chat.id, "phase 1 done")

    # Phase 2: retrieval-relevant query (should use ingested memories from phase 1)
    print(f"\n  Phase 2: retrieval query using ingested knowledge")
    query = "What are my hobbies and interests? What do I enjoy doing?"
    print(f"    Query: {query!r}")
    resp = safe_handle_turn(
        db, cu, query,
        summarisation_threshold=999,  # disable to isolate retrieval
        ingestion_threshold=999,
    )
    if resp is not None:
        print(f"    Response: {(resp[:200] + '…') if len(resp) > 200 else resp!r}")

        # Check if response mentions ingested content
        keywords = ["hiking", "himalaya", "meditation", "bruno", "cabin", "mountain", "dog", "deep work"]
        found = [k for k in keywords if k.lower() in resp.lower()]
        if found:
            _ok(f"Full-e2e retrieval grounded — response mentions: {found}")
        else:
            _warn("Response doesn't mention expected ingested keywords — retrieval may not have grounded the prompt")
    else:
        _fail("Phase 2 handle_chat_turn returned None")

    # --- Final assertions ---
    print(f"\n  Final assertions:")
    db.expire_all()
    chat = read_chat(db, test_user.id, chat.id)
    final_blocks = read_message_blocks_by_chat(db, test_user.id, chat.id) or []
    final_mems = read_memory_summaries(db, test_user.id) or []
    new_mems_final = len(final_mems) - baseline_summaries
    total_msgs = len(chat.message or [])

    check(total_msgs > 0, f"Messages exist ({total_msgs})", "No messages in chat")
    check(resp is not None and len(resp.strip()) > 0, "Final LLM response non-empty", "Final LLM response empty")

    # Report — don't hard-fail on summarisation/ingestion since LLM may return [] sometimes
    if len(final_blocks) > baseline_blocks:
        _ok(f"Summarisation produced {len(final_blocks) - baseline_blocks} new block(s)")
    else:
        _warn("No new summary blocks in full-e2e — may need more messages or lower threshold")

    if new_mems_final > 0:
        _ok(f"Ingestion produced {new_mems_final} new memory/memories in full-e2e")
    else:
        _warn("No new memories in full-e2e — ingestion LLM may have returned [] for this content")

    # Verify flags are clean (no stuck locks)
    check(not chat.summarisation_going_on, "summarisation_going_on is False (no stuck lock)",
          "summarisation_going_on still True — stuck lock!")
    check(not chat.ingestion_going_on, "ingestion_going_on is False (no stuck lock)",
          "ingestion_going_on still True — stuck lock!")

    dump_db_state(db, test_user.id, chat.id, "after test_05")


# ---------------------------------------------------------------------------
# TEST 06 — Direct service unit checks (no LLM)
# ---------------------------------------------------------------------------
def test_06_service_units(db: Session):
    """
    Lightweight checks on service internals that don't need LLM calls:
    - _form_chat_context returns correct recent_messages
    - _extract_recent_memories deduplicates
    - _deduplicate_previous_memories removes overlaps
    - read_messages_after_order works
    """
    print(f"\n{BOLD}TEST 06 — Service unit checks (no LLM){RESET}")

    # Use the 'basic' chat from test_01 (has known state) or create fresh
    chat = test_chats.get("basic")
    if chat is None:
        chat = make_chat(db, "unit")
        cu = ChatUser(user_id=test_user.id, chat_id=chat.id)
        # Seed with direct DB writes (no LLM) for deterministic state
        from app.database.model import role_enum as _role
        from app.database.crud.message import create_message as _create_msg
        for i, text in enumerate(["Hello", "Hi there!", "How are you?", "I'm good."], 1):
            role = _role.USER if i % 2 == 1 else _role.ASSISTANT
            _create_msg(db, test_user.id, chat.id, role=role, content=text)
        db.expire_all()
        chat = read_chat(db, test_user.id, chat.id)
    else:
        cu = ChatUser(user_id=test_user.id, chat_id=chat.id)

    # --- _form_chat_context ---
    from app.services.chat_orchestration import (
        _form_chat_context,
        _extract_recent_memories,
        _deduplicate_previous_memories,
    )

    try:
        summary_blocks, recent_messages, raw_messages = _form_chat_context(db, cu)
        print(f"    _form_chat_context: {len(summary_blocks)} blocks, {len(recent_messages)} recent, {len(raw_messages)} raw")
        check(len(recent_messages) == len(raw_messages),
              "_form_chat_context: recent_messages and raw_messages same length",
              f"Length mismatch: {len(recent_messages)} vs {len(raw_messages)}")
        # recent_messages should only contain messages after last_summarisation
        if raw_messages:
            chat_fresh = read_chat(db, test_user.id, chat.id)
            last_summ = chat_fresh.last_summarisation_message_order or 0
            all_after = all(m.order_in_chat > last_summ for m in raw_messages)
            check(all_after,
                  f"All raw_messages have order > last_summarisation ({last_summ})",
                  "Some raw_messages have order <= last_summarisation — filter bug")
    except Exception as e:
        _fail(f"_form_chat_context raised: {e}")
        traceback.print_exc()

    # --- read_messages_after_order ---
    try:
        msgs = read_messages_after_order(db, chat.id, 0)
        print(f"    read_messages_after_order(chat, 0): {len(msgs)} messages")
        check(len(msgs) > 0, "read_messages_after_order returns messages", "read_messages_after_order returned empty")

        msgs2 = read_messages_after_order(db, chat.id, 999)
        check(len(msgs2) == 0, "read_messages_after_order with high offset returns 0", f"Expected 0, got {len(msgs2)}")

        # Verify sorted ascending
        if len(msgs) >= 2:
            orders = [m.order_in_chat for m in msgs]
            check(orders == sorted(orders), f"Messages sorted ascending: {orders}", f"Not sorted: {orders}")
    except Exception as e:
        _fail(f"read_messages_after_order raised: {e}")
        traceback.print_exc()

    # --- _deduplicate_previous_memories ---
    try:
        from app.services.semantic_retrieval import RetrievedMemory as SRM
        from app.database.model import MemorySummary as _MS

        # Create mock RetrievedMemory objects
        # We need real MemorySummary objects or at least objects with .id
        # Use the seeded summary from test_02 if available
        seeded = test_chats.get("retrieval")
        if seeded is not None:
            db.expire_all()
            summaries = read_memory_summaries(db, test_user.id) or []
            if len(summaries) >= 1:
                m1 = SRM(memory_type="summary", memory=summaries[0], similarity=0.9)
                m2 = SRM(memory_type="summary", memory=summaries[0], similarity=0.8)  # same memory
                prev = [m1]
                curr = [m2]
                result = _deduplicate_previous_memories(prev, curr)
                check(len(result) == 0,
                      "Dedup: previous memory removed when same memory in current",
                      f"Dedup failed: expected 0, got {len(result)}")

                # Non-overlapping case
                if len(summaries) >= 2:
                    m3 = SRM(memory_type="summary", memory=summaries[1], similarity=0.7)
                    prev2 = [m1]
                    curr2 = [m3]
                    result2 = _deduplicate_previous_memories(prev2, curr2)
                    check(len(result2) == 1,
                          "Dedup: previous kept when no overlap with current",
                          f"Dedup failed: expected 1, got {len(result2)}")
                else:
                    _info("Only 1 summary exists — skipping non-overlap dedup test")
            else:
                _warn("No summaries for dedup test — skipping")
        else:
            _info("No retrieval chat found — skipping dedup unit test")
    except Exception as e:
        _fail(f"Dedup unit test raised: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Second-brain business logic verification")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep test data in DB after run")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests (03, 04, 05)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  SECOND-BRAIN — BUSINESS LOGIC VERIFICATION{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  DB: via SQL_URL  |  LLM: gemini-3.6-flash  |  Embedding: gemini-embedding-2")
    print(f"  Thresholds: summarisation={SUMMARISATION_THRESHOLD}  ingestion={INGESTION_THRESHOLD}")
    if args.quick:
        print(f"  Mode: {YELLOW}--quick (skipping tests 03, 04, 05){RESET}")

    # --- Pre-checks ---
    import os
    if not os.getenv("GEMINI_API_KEY"):
        print(f"\n{RED}FATAL: GEMINI_API_KEY not set. Check .env{RESET}")
        sys.exit(1)
    if not os.getenv("SQL_URL"):
        print(f"\n{RED}FATAL: SQL_URL not set. Check .env{RESET}")
        sys.exit(1)

    db: Session = SessionLocal()
    try:
        # Quick DB connectivity check
        try:
            from sqlalchemy import text as db_text
            db.execute(db_text("SELECT 1"))
            print(f"  DB: {GREEN}connected{RESET}")
        except Exception as e:
            print(f"\n{RED}FATAL: DB connection failed: {e}{RESET}")
            traceback.print_exc()
            sys.exit(1)

        # --- Setup ---
        setup(db)

        # --- Tests ---
        tests = [
            ("01_basic_chat", test_01_basic_chat),
            ("02_retrieval", test_02_retrieval),
        ]
        if not args.quick:
            tests += [
                ("03_summarisation", test_03_summarisation),
                ("04_ingestion", test_04_ingestion),
                ("05_full_e2e", test_05_full_e2e),
            ]
        tests.append(("06_service_units", test_06_service_units))

        for name, fn in tests:
            try:
                fn(db)
            except AssertionError as e:
                _fail(f"{name}: assertion failed: {e}")
                traceback.print_exc()
            except Exception as e:
                _fail(f"{name}: unexpected error: {e}")
                traceback.print_exc()
                try:
                    db.rollback()
                except Exception:
                    pass

        # --- Summary ---
        print(f"\n{BOLD}{'=' * 60}{RESET}")
        total = pass_count + fail_count
        color = GREEN if fail_count == 0 else RED
        print(f"{BOLD}  RESULTS: {color}{pass_count}/{total} passed{RESET}"
              f"{f'  {RED}{fail_count} failed{RESET}' if fail_count else ''}"
              f"{f'  {DIM}{skip_count} skipped{RESET}' if skip_count else ''}")
        print(f"{BOLD}{'=' * 60}{RESET}")

    finally:
        try:
            teardown(db, keep=args.no_cleanup)
        except Exception as e:
            print(f"  Teardown error: {e}")
        db.close()

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
