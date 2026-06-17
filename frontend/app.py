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

# Auto-generated user/session IDs
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{str(uuid.uuid4())[:8]}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 Mnemosyne AI Agent")
st.caption("Persistent memory AI agent")

# Sidebar for new chat/refresh controls
with st.sidebar:
    st.header("Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = f"session_{str(uuid.uuid4())[:8]}"
            st.experimental_rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.experimental_rerun()

    # Show current session info (optional, for debugging)
    with st.expander("Session Info"):
        st.text(f"User ID: {st.session_state.user_id}")
        st.text(f"Session ID: {st.session_state.session_id}")

    # Conversation history
    st.subheader("Conversation History")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            # Truncate content for brevity
            content_preview = msg["content"].replace("\n", " ")[:100]
            st.caption(f"{role}: {content_preview}...")
    else:
        st.caption("No conversation yet.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request to backend
    turn_data = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "memories": [],  # No explicit memories from UI for now
        "query": prompt,
        "top_k": 5,
        "history": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]  # exclude current user message
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

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

# Footer
st.divider()
st.caption("Mnemosyne AI Agent - Powered by persistent memory")