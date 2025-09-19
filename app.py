import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.3-2b-instruct"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_hf_api(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # The response might be a list of dicts with 'generated_text'
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return "Unexpected response format."
    else:
        return f"Error {response.status_code}: {response.text}"

def main():
    st.title("IBM Granite Chatbot (via Hugging Face API)")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:", key="input")

    if st.button("Send") and user_input:
        # Add user input to chat history
        st.session_state.history.append({"role": "user", "content": user_input})

        # Build prompt as conversation text to keep context (optional)
        conversation = ""
        for msg in st.session_state.history:
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n"
            else:
                conversation += f"Assistant: {msg['content']}\n"
        conversation += "Assistant:"

        # Query the model
        response = query_hf_api(conversation)

        # Save assistant's response to history
        st.session_state.history.append({"role": "assistant", "content": response})

    # Display chat history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

if __name__ == "__main__":
    main()
