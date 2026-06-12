"""
tests/test_mnemosyne.py

Full integration test suite for Mnemosyne.
Tests the live API running at http://localhost:8000.

Run with:
    python tests/test_mnemosyne.py

Or verbosely:
    python tests/test_mnemosyne.py -v
"""

from __future__ import annotations
import sys
import json
import time
import asyncio
import argparse
import httpx
from datetime import datetime

BASE_URL  = "http://localhost:8000"
TEST_USER = f"test_user_{int(time.time())}"   # unique per run to avoid state bleed

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


# Helpers 

def log(label: str, passed: bool, detail: str = ""):
    results.append((label, passed, detail))
    icon = PASS if passed else FAIL
    print(f"  {icon}  {label}")
    if detail and (not passed or "-v" in sys.argv):
        for line in detail.strip().splitlines():
            print(f"       {INFO} {line}")


def section(title: str):
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


async def post(client: httpx.AsyncClient, path: str, body: dict) -> dict:
    r = await client.post(f"{BASE_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


async def get(client: httpx.AsyncClient, path: str) -> dict:
    r = await client.get(f"{BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


# Test Groups 

async def test_health(client: httpx.AsyncClient):
    section("1. Health Check")
    try:
        data = await get(client, "/health")
        log("API is reachable", True, f"version={data.get('version')}")
        log("Status is ok", data.get("status") == "ok")
    except Exception as e:
        log("API is reachable", False, str(e))


async def test_memory_write(client: httpx.AsyncClient) -> str:
    """Returns session_id for reuse in subsequent tests."""
    section("2. Memory Write")
    session_id = None

    memories = [
        {"content": "User prefers dark mode",           "memory_type": "preference"},
        {"content": "User is a software engineer",      "memory_type": "fact"},
        {"content": "User asked about Python async IO", "memory_type": "episode"},
        {"content": "Always give concise answers",      "memory_type": "rule"},
        {"content": "User loves coffee",                "memory_type": "preference"},
        {"content": "User works at a fintech company",  "memory_type": "fact"},
    ]

    try:
        data = await post(client, "/turn", {
            "user_id":  TEST_USER,
            "memories": memories,
            "query":    "",
        })

        session_id = data.get("session_id")
        written    = data.get("written", [])

        log("Turn returns session_id",   bool(session_id),         f"session_id={session_id}")
        log("Turn returns turn number",  "turn" in data,           f"turn={data.get('turn')}")
        log(f"All {len(memories)} memories written",
            len(written) == len(memories),
            f"written={len(written)}")

    except Exception as e:
        log("Memory write via /turn", False, str(e))

    return session_id


async def test_direct_write(client: httpx.AsyncClient):
    section("3. Direct /memory/write")
    try:
        data = await post(client, "/memory/write", {
            "user_id":     TEST_USER,
            "content":     "User never wants email notifications",
            "memory_type": "rule",
            "tags":        ["notifications", "email"],
        })
        log("Direct write succeeds",      data.get("status") == "written")
        log("Returns memory_id",          bool(data.get("memory_id")),
            f"memory_id={data.get('memory_id')}")
    except Exception as e:
        log("Direct /memory/write", False, str(e))


async def test_memory_list(client: httpx.AsyncClient):
    section("4. Memory List")
    try:
        data = await get(client, f"/memory/list/{TEST_USER}")
        count     = data.get("count", 0)
        memories  = data.get("memories", [])

        log("List endpoint responds",     True)
        log("Returns correct user_id",    data.get("user_id") == TEST_USER)
        log("Has memories in DB",         count > 0, f"count={count}")

        if memories:
            sample = memories[0]
            log("Memory has required fields", all(k in sample for k in [
                "id", "user_id", "content", "memory_type",
                "importance_score", "status", "turns_since_access"
            ]))

            types_found = {m["memory_type"] for m in memories}
            log("All 4 memory types present",
                types_found >= {"preference", "fact", "episode", "rule"},
                f"found={types_found}")

    except Exception as e:
        log("Memory list", False, str(e))


async def test_retrieval(client: httpx.AsyncClient, session_id: str):
    section("5. Memory Retrieval")

    queries = [
        ("dark mode",          "preference"),
        ("software engineer",  "fact"),
        ("Python async",       "episode"),
        ("concise answers",    "rule"),
    ]

    for query, expected_type in queries:
        try:
            data = await post(client, "/memory/retrieve", {
                "user_id": TEST_USER,
                "query":   query,
                "top_k":   3,
            })
            results_list = data.get("results", [])
            has_results  = len(results_list) > 0

            # Check top result is relevant
            top_type = None
            if results_list:
                top_type = results_list[0]["payload"].get("memory_type")

            log(f"Query '{query}' returns results",
                has_results,
                f"top_result_type={top_type}, score={results_list[0]['score']:.3f}" if results_list else "no results")

        except Exception as e:
            log(f"Query '{query}'", False, str(e))


async def test_turn_with_reply(client: httpx.AsyncClient, session_id: str):
    section("6. /turn with Qwen Reply")

    queries = [
        "What display settings do I prefer?",
        "What do you know about my job?",
        "How should you format your responses to me?",
    ]

    for query in queries:
        try:
            data = await post(client, "/turn", {
                "user_id":    TEST_USER,
                "session_id": session_id,
                "memories":   [],
                "query":      query,
                "top_k":      3,
            })

            reply     = data.get("reply", "")
            retrieved = data.get("retrieved", [])

            log(f"Query returns reply",
                bool(reply),
                f"query='{query[:40]}'\n"
                f"retrieved={len(retrieved)} memories\n"
                f"reply='{reply[:120]}...' " if len(reply) > 120 else f"reply='{reply}'")

        except Exception as e:
            log(f"Turn with query '{query[:40]}'", False, str(e))


async def test_multi_turn_memory(client: httpx.AsyncClient):
    section("7. Multi-Turn Memory Persistence")

    user = f"multi_turn_{int(time.time())}"

    try:
        # Turn 1: introduce a fact
        t1 = await post(client, "/turn", {
            "user_id":  user,
            "memories": [{"content": "My name is Newton", "memory_type": "fact"}],
            "query":    "Hello",
        })
        t1_turn = t1.get("turn")
        session_id = t1.get("session_id")
        log("Turn 1 created", t1_turn == 1, f"turn={t1_turn}")

        # Turn 2: ask something that requires the memory
        t2 = await post(client, "/turn", {
            "user_id":  user,
            "session_id": session_id,
            "memories": [],
            "query":    "What is my name?",
            "history":  [
                {"role": "user",      "content": "Hello"},
                {"role": "assistant", "content": t1.get("reply", "")},
            ],
        })

        reply     = t2.get("reply", "")
        retrieved = t2.get("retrieved", [])
        turn_2    = t2.get("turn")

        log("Turn 2 increments correctly", turn_2 == 2, f"turn={turn_2}")
        log("Turn 2 retrieves prior memory", len(retrieved) > 0,
            f"retrieved={len(retrieved)} memories")
        log("Reply references the name",
            "newton" in reply.lower(),
            f"reply='{reply}'")

    except Exception as e:
        log("Multi-turn memory", False, str(e))


async def test_importance_scoring(client: httpx.AsyncClient):
    section("8. Importance Score Estimation")

    test_cases = [
        ("always use markdown formatting",  "rule",       0.85),
        ("user is 30 years old",            "fact",       0.60),
        ("user asked about the weather",    "episode",    0.40),
        ("user prefers dark themes",        "preference", 0.70),
        ("never send notifications",        "rule",       0.85),  # 'never' boost
    ]

    for content, mtype, expected_min in test_cases:
        try:
            data = await post(client, "/memory/write", {
                "user_id":     TEST_USER,
                "content":     content,
                "memory_type": mtype,
            })
            mid = data.get("memory_id")

            # Fetch the record
            list_data = await get(client, f"/memory/list/{TEST_USER}")
            mem = next(
                (m for m in list_data["memories"] if m["id"] == mid), None
            )

            if mem:
                score = mem.get("importance_score", 0)
                log(f"'{content[:35]}...' importance >= {expected_min}",
                    score >= expected_min,
                    f"score={score:.2f}, type={mtype}")
            else:
                log(f"Could not fetch memory {mid}", False)

        except Exception as e:
            log(f"Importance for '{content[:35]}'", False, str(e))


async def test_memory_aging(client: httpx.AsyncClient):
    section("9. Memory Aging (turns_since_access)")

    user = f"aging_{int(time.time())}"

    try:
        # Write a memory
        await post(client, "/memory/write", {
            "user_id":     user,
            "content":     "User likes jazz music",
            "memory_type": "preference",
        })

        # Fire 3 turns without accessing that memory
        for _ in range(3):
            await post(client, "/turn", {
                "user_id":  user,
                "memories": [],
                "query":    "",
            })

        # Check turns_since_access
        data = await get(client, f"/memory/list/{user}")
        mems = data.get("memories", [])

        if mems:
            age = mems[0].get("turns_since_access", 0)
            log("turns_since_access increments per turn",
                age >= 3,
                f"turns_since_access={age} (expected ≥ 3)")
        else:
            log("Memory aging", False, "No memories found")

    except Exception as e:
        log("Memory aging", False, str(e))


async def test_graph_endpoint(client: httpx.AsyncClient):
    section("10. Memory Graph")

    try:
        data = await get(client, f"/memory/graph/{TEST_USER}")
        nodes = data.get("nodes", [])
        edges = data.get("edges", {})

        log("Graph endpoint responds",  True)
        log("Graph has nodes",          len(nodes) > 0, f"nodes={len(nodes)}")
        log("Graph has edge structure", isinstance(edges, dict),
            f"edges for {len(edges)} nodes")

        # Count total edges
        total_edges = sum(len(v) for v in edges.values())
        log("Semantic edges exist between similar memories",
            total_edges > 0,
            f"total_edge_connections={total_edges}")

    except Exception as e:
        log("Memory graph", False, str(e))


async def test_session_list(client: httpx.AsyncClient):
    section("11. Session Management")

    try:
        data = await get(client, f"/sessions/{TEST_USER}")
        sessions = data.get("sessions", [])

        log("Sessions endpoint responds",  True)
        log("Has at least one session",    len(sessions) > 0,
            f"sessions={len(sessions)}")

    except Exception as e:
        log("Session list", False, str(e))


# Summary 

def print_summary():
    section("RESULTS")
    passed = sum(1 for _, p, _ in results if p)
    total  = len(results)
    failed = [(name, detail) for name, p, detail in results if not p]

    print(f"\n  {PASS} Passed: {passed}/{total}")

    if failed:
        print(f"  {FAIL} Failed: {len(failed)}/{total}\n")
        print(f"  {BOLD}Failed tests:{RESET}")
        for name, detail in failed:
            print(f"    {FAIL} {name}")
            if detail:
                print(f"         {detail}")
    else:
        print(f"\n  {BOLD}All tests passed! Mnemosyne is healthy.{RESET}")

    print()
    return len(failed) == 0


# Runner 

async def run():
    print(f"\n{BOLD}MNEMOSYNE INTEGRATION TESTS{RESET}")
    print(f"  Target : {BASE_URL}")
    print(f"  User   : {TEST_USER}")
    print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    async with httpx.AsyncClient() as client:
        await test_health(client)

        session_id = await test_memory_write(client)

        await test_direct_write(client)
        await test_memory_list(client)
        await test_retrieval(client, session_id)
        await test_turn_with_reply(client, session_id)
        await test_multi_turn_memory(client)
        await test_importance_scoring(client)
        await test_memory_aging(client)
        await test_graph_endpoint(client)
        await test_session_list(client)

    return print_summary()


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)