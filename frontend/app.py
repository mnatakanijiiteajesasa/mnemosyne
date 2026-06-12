import streamlit as st
import requests
import os
import json
from datetime import datetime

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Mnemosyne AI Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mnemosyne AI Agent")
st.caption("Persistent memory AI agent")

# Sidebar for user/session configuration
with st.sidebar:
    st.header("Configuration")
    user_id = st.text_input("User ID", value="demo_user")
    session_id = st.text_input("Session ID (optional)", value="")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        "user_id": user_id,
        "session_id": session_id if session_id else None,
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