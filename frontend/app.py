import streamlit as st
import requests
import os
import json
from datetime import datetime
import uuid
import re

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Mnemosyne AI Agent",
    page_icon="🧠",
    layout="wide"
)


def generate_topic_code(messages):
    """
    Generate a 3-4 letter topic code from session messages.
    Extracts significant words and returns their initials as an uppercase code.
    """
    if not messages:
        return "???"

    # Combine all user messages (or all messages if preferred)
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    if not user_messages:
        # Fallback to all messages if no user messages
        all_messages = [msg["content"] for msg in messages]
        text = " ".join(all_messages)
    else:
        text = " ".join(user_messages)

    if not text.strip():
        return "???"

    # Simple stopwords list (common English words to ignore)
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'which',
        'their', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'we',
        'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so',
        'some', 'her', 'would', 'make', 'like', 'into', 'has', 'more', 'go', 'no',
        'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'oe', 'its',
        'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may',
        'part'
    }

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

    # Filter out stopwords and get unique words
    meaningful_words = [w for w in words if w not in stopwords]

    # If we don't have enough meaningful words, fall back to all words
    if len(meaningful_words) < 3:
        meaningful_words = words

    # Sort by length (longer words first) to get more significant terms
    meaningful_words.sort(key=len, reverse=True)

    # Take first letters of up to 4 words
    initials = []
    for word in meaningful_words[:4]:
        if word:  # Ensure word is not empty
            initials.append(word[0].upper())

    # Ensure we have at least 3 characters
    while len(initials) < 3:
        initials.append('X')  # Padding character

    # Take exactly 3-4 characters
    result = ''.join(initials[:4])

    # If we somehow got less than 3, pad with X
    if len(result) < 3:
        result = result.ljust(3, 'X')

    return result

# Auto-generated user ID
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"

# Initialize sessions list and current session
if "sessions" not in st.session_state:
    # Create first session
    initial_session = {
        "session_id": f"session_{str(uuid.uuid4())[:8]}",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.sessions = [initial_session]
    st.session_state.current_session_id = initial_session["session_id"]

def get_current_session():
    """Return the current session object."""
    for sess in st.session_state.sessions:
        if sess["session_id"] == st.session_state.current_session_id:
            return sess
    # Fallback (should not happen)
    return st.session_state.sessions[0] if st.session_state.sessions else None

st.title("Mnemosyne AI Agent")
st.caption("Persistent memory AI agent")

# Sidebar for controls and history
with st.sidebar:
    st.header("Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True):
            # Create new session
            new_session = {
                "session_id": f"session_{str(uuid.uuid4())[:8]}",
                "messages": [],
                "created_at": datetime.now()
            }
            st.session_state.sessions.append(new_session)
            st.session_state.current_session_id = new_session["session_id"]
            st.experimental_rerun()
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.experimental_rerun()

    # Show current session info (optional, for debugging)
    with st.expander("Session Info"):
        st.text(f"User ID: {st.session_state.user_id}")
        current = get_current_session()
        if current:
            st.text(f"Session ID: {current['session_id']}")
            st.text(f"Messages: {len(current['messages'])}")
        else:
            st.text("Session ID: None")

    # Session history
    st.subheader("Session History")
    if st.session_state.sessions:
        # Sort sessions by creation time (newest first)
        sorted_sessions = sorted(st.session_state.sessions, key=lambda x: x["created_at"], reverse=True)
        for sess in sorted_sessions:
            # Determine if this is the current session
            is_current = sess["session_id"] == st.session_state.current_session_id
            # Create a label
            label = "🟢 Current" if is_current else ""
            # Generate topic summary (3-4 letters) from session messages
            topic_code = generate_topic_code(sess["messages"])
            # Display session as clickable button
            button_label = f"{topic_code} {label}".strip()
            if st.button(button_label, key=f"session_{sess['session_id']}", use_container_width=True):
                if not is_current:  # Only switch if it's not already current
                    st.session_state.current_session_id = sess["session_id"]
                    st.experimental_rerun()
    else:
        st.caption("No sessions yet.")

# Display chat messages from current session
current_session = get_current_session()
if current_session:
    for message in current_session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's poppin'?"):
    # Add user message to current session
    current_session = get_current_session()
    if current_session:
        current_session["messages"].append({"role": "user", "content": prompt})
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare request to backend
        turn_data = {
            "user_id": st.session_state.user_id,
            "session_id": current_session["session_id"],
            "memories": [],  # No explicit memories from UI for now
            "query": prompt,
            "top_k": 5,
            "history": [{"role": m["role"], "content": m["content"]} for m in current_session["messages"][:-1]]  # exclude current user message
        }

        # Show thinking spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/turn", json=turn_data, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    assistant_reply = result.get("reply", "Sorry, I couldn't generate a response.")

                    # Show additional info in expander
                    with st.expander("Debug info"):
                        st.json({
                            "session_id": result.get("session_id"),
                            "turn": result.get("turn"),
                            "written": result.get("written"),
                            "retrieved_count": len(result.get("retrieved", [])),
                            "archived": result.get("archived")
                        })

                except requests.exceptions.RequestException as e:
                    assistant_reply = f"Error communicating with backend: {str(e)}"
                except Exception as e:
                    assistant_reply = f"Unexpected error: {str(e)}"

        # Add assistant response to current session
        current_session["messages"].append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

# Footer
st.divider()
st.caption("Mnemosyne AI Agent - Powered by persistent will")