#!/usr/bin/env python3
"""
Stress test for Mnemosyne persistent memory system.
Tests that memories persist and are retrieved correctly across conversation turns.
"""

import asyncio
import httpx
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_USER = f"stress_test_user_{int(time.time())}"

async def test_persistent_memory():
    """Test that memories persist across turns and are retrieved contextually."""

    print(f"\n{'='*60}")
    print(f"Mnemosyne Persistent Memory Stress Test")
    print(f"{'='*60}")
    print(f"Test User: {TEST_USER}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Test scenarios
    test_scenarios = [
        {
            "name": "Preference Memory Test",
            "setup": [
                {"content": "I prefer dark mode for all applications", "memory_type": "preference"},
                {"content": "I dislike pop-up advertisements", "memory_type": "preference"}
            ],
            "queries": [
                "What display mode do I prefer?",
                "Do I like advertisements?",
                "Should I enable dark mode in my editor?"
            ],
            "expected_keywords": [["dark", "mode"], ["dislike", "advertisement"], ["dark", "mode"]]
        },
        {
            "name": "Fact Memory Test",
            "setup": [
                {"content": "I work as a machine learning engineer at TechCorp", "memory_type": "fact"},
                {"content": "I live in San Francisco, California", "memory_type": "fact"},
                {"content": "I graduated from MIT with a degree in Computer Science", "memory_type": "fact"}
            ],
            "queries": [
                "What is my profession?",
                "Where do I live?",
                "Where did I go to college?"
            ],
            "expected_keywords": [["machine", "learning", "engineer"], ["San", "Francisco"], ["MIT", "Computer", "Science"]]
        },
        {
            "name": "Episode Memory Test",
            "setup": [
                {"content": "Yesterday I debugged a complex neural network issue involving vanishing gradients", "memory_type": "episode"},
                {"content": "Last Tuesday I presented my research findings at the AI conference", "memory_type": "episode"},
                {"content": "This morning I pair-programmed with a junior developer on a authentication bug", "memory_type": "episode"}
            ],
            "queries": [
                "What did I debug yesterday?",
                "When did I present at the AI conference?",
                "What did I work on this morning?"
            ],
            "expected_keywords": [["debugged", "neural", "network"], ["Tuesday", "conference"], ["morning", "pair-programmed"]]
        },
        {
            "name": "Rule Memory Test",
            "setup": [
                {"content": "Always respond with code examples when explaining programming concepts", "memory_type": "rule"},
                {"content": "Never share personal identification information in responses", "memory_type": "rule"},
                {"content": "Prefer concise answers unless thorough explanation is requested", "memory_type": "rule"}
            ],
            "queries": [
                "How should I explain programming concepts?",
                "Should I share my SSN if asked?",
                "Do you prefer brief or detailed answers?"
            ],
            "expected_keywords": [["code", "examples"], ["never", "share"], ["concise"]]
        }
    ]

    all_passed = True

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Health check first
        try:
            health_resp = await client.get(f"{BASE_URL}/health")
            health_data = health_resp.json()
            print(f"✓ Backend Health: {health_data['status']} (v{health_data['version']})")
        except Exception as e:
            print(f"✗ Backend Health Check Failed: {e}")
            return False

        # Run each test scenario
        for scenario_idx, scenario in enumerate(test_scenarios, 1):
            print(f"\n{scenario_idx}. {scenario['name']}")
            print("-" * 50)

            # Setup phase: Write memories
            print("  Setting up memories...")
            setup_success = True

            for memory in scenario['setup']:
                try:
                    resp = await client.post(
                        f"{BASE_URL}/memory/write",
                        json={
                            "user_id": TEST_USER,
                            "content": memory["content"],
                            "memory_type": memory["memory_type"]
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        print(f"    ✓ Written: {memory['content'][:50]}{'...' if len(memory['content']) > 50 else ''}")
                    else:
                        print(f"    ✗ Failed to write: {memory['content'][:50]}...")
                        setup_success = False
                except Exception as e:
                    print(f"    ✗ Error writing memory: {e}")
                    setup_success = False

            if not setup_success:
                print(f"  ✗ Scenario setup failed")
                all_passed = False
                continue

            # Wait a moment for processing
            await asyncio.sleep(0.5)

            # Testing phase: Ask queries and check responses
            print("  Testing retrieval...")
            scenario_passed = True

            for query_idx, (query, expected_keywords) in enumerate(zip(scenario['queries'], scenario['expected_keywords']), 1):
                try:
                    # Get conversation history for context
                    history_resp = await client.get(f"{BASE_URL}/sessions/{TEST_USER}")
                    history_data = history_resp.json()

                    # Prepare turn data
                    turn_data = {
                        "user_id": TEST_USER,
                        "query": query,
                        "top_k": 5
                    }

                    # Add session_id if we have sessions
                    if history_data.get("sessions"):
                        turn_data["session_id"] = history_data["sessions"][0]["id"]

                    # Execute the turn
                    resp = await client.post(f"{BASE_URL}/turn", json=turn_data)

                    if resp.status_code == 200:
                        data = resp.json()
                        reply = data.get("reply", "").lower()

                        # Check if expected keywords appear in reply
                        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in reply)
                        keyword_ratio = keywords_found / len(expected_keywords)

                        if keyword_ratio >= 0.5:  # At least half the keywords found
                            print(f"    ✓ Q{query_idx}: '{query}'")
                            print(f"      Reply: '{reply[:100]}{'...' if len(reply) > 100 else ''}'")
                        else:
                            print(f"    ⚠ Q{query_idx}: '{query}'")
                            print(f"      Reply: '{reply[:100]}{'...' if len(reply) > 100 else ''}'")
                            print(f"      Expected keywords: {expected_keywords}")
                            print(f"      Keywords found: {[kw for kw in expected_keywords if kw.lower() in reply]}")
                            scenario_passed = False
                    else:
                        print(f"    ✗ Q{query_idx}: HTTP {resp.status_code}")
                        scenario_passed = False

                except Exception as e:
                    print(f"    ✗ Q{query_idx}: Error - {e}")
                    scenario_passed = False

                # Small delay between queries
                await asyncio.sleep(0.3)

            if scenario_passed:
                print(f"  ✓ Scenario PASSED")
            else:
                print(f"  ✗ Scenario FAILED")
                all_passed = False

    # Final summary
    print(f"\n{'='*60}")
    if all_passed:
        print(f"🎉 ALL TESTS PASSED - Persistent memory system is working correctly!")
        print(f"   Memories are being stored, retrieved, and contextualized properly.")
    else:
        print(f"❌ SOME TESTS FAILED - There may be issues with memory persistence.")
    print(f"{'='*60}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_passed

if __name__ == "__main__":
    result = asyncio.run(test_persistent_memory())
    exit(0 if result else 1)