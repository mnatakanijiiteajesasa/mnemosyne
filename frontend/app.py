import streamlit as st
import requests
import os
import json
from datetime import datetime
import uuid

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Mnemosyne AI Agent",
    page_icon="🧠",
    layout="wide"
)

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
            label = "Current" if is_current else ""
            # Preview: first user message or empty
            preview = "Empty session"
            if sess["messages"]:
                # Find first user message
                for msg in sess["messages"]:
                    if msg["role"] == "user":
                        preview = msg["content"].replace("\n", " ")[:60]
                        break
                else:
                    # No user message, maybe first is assistant
                    preview = sess["messages"][0]["content"].replace("\n", " ")[:60]
            # Display session as clickable button
            button_label = f"{preview}... {label}".strip()
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
if prompt := st.chat_input("What's on your mind?"):
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
st.caption("Mnemosyne AI Agent - Powered by persistent memory")