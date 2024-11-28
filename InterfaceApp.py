import streamlit as st
from PEFTChat import PEFTChatModel

# Initialize the chatbot model
@st.cache_resource
def initialize_model():
    return PEFTChatModel(
        base_model_path="gpt2",  # Replace with your base model path
        peft_model_path="./FinetunedModels/gpt2_finetuned",  # Replace with your fine-tuned model path
        knowledge_address="./dataset/knowledge-base.md",
        livedataset_address="./dataset/live-datasource.json",
        max_new_tokens=20,
        temperature=0.5,
    )

# Streamlit app layout
st.title("PEFT Chatbot")
st.write("This chatbot uses PEFT fine-tuned models to answer your queries.")

# Initialize model
chat_model = initialize_model()

# Session state to manage conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Input box for user message
user_message = st.text_input("Your Message", placeholder="Type your question here...")

# Generate response on button click
if st.button("Send"):
    if user_message.strip():
        try:
            # Generate response
            response = chat_model.generate_response(user_message)

            # Append to conversation history
            st.session_state.conversation_history.append(("Customer", user_message))
            st.session_state.conversation_history.append(("Agent", response))
        except Exception as e:
            st.error(f"Error: {e}")

# Display conversation history
if st.session_state.conversation_history:
    st.write("### Conversation History:")
    for sender, message in st.session_state.conversation_history:
        if sender == "Customer":
            st.write(f"**Customer:** {message}")
        else:
            st.write(f"**Agent:** {message}")

# Clear history button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
