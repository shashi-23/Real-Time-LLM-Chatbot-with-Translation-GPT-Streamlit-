# chat_app.py

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from deep_translator import GoogleTranslator

# Load environment variables from .env
load_dotenv(find_dotenv(), override=False)

# Language options
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW",
    "Japanese": "ja", "Korean": "ko", "Hindi": "hi", "Arabic": "ar",
    "Russian": "ru", "Portuguese": "pt", "Italian": "it", "Dutch": "nl",
    "Turkish": "tr", "Vietnamese": "vi", "Polish": "pl", "Swedish": "sv",
    "Norwegian": "no", "Finnish": "fi", "Czech": "cs", "Greek": "el",
    "Hungarian": "hu", "Thai": "th", "Hebrew": "he", "Indonesian": "id",
    "Malay": "ms", "Ukrainian": "uk", "Romanian": "ro", "Slovak": "sk",
    "Danish": "da", "Bulgarian": "bg", "Croatian": "hr", "Serbian": "sr",
    "Lithuanian": "lt", "Latvian": "lv", "Slovenian": "sl", "Estonian": "et",
    "Filipino": "tl", "Icelandic": "is", "Basque": "eu", "Galician": "gl",
    "Welsh": "cy", "Irish": "ga", "Swahili": "sw", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Punjabi": "pa",
    "Urdu": "ur"
}

# Sidebar: Language selector
st.sidebar.title("⚙️ Settings")
selected_language = st.sidebar.selectbox("🌐 Language:", list(LANGUAGES.keys()))
selected_lang_code = LANGUAGES[selected_language]

# Sidebar: Chat history summary
st.sidebar.markdown("---")
st.sidebar.markdown("📜 **Chat History**")
if "conversation_log" in st.session_state:
    for idx, turn in enumerate(st.session_state.conversation_log):
        st.sidebar.markdown(f"**Q{idx+1}:** {turn['user'][:30]}...")

# Initialize OpenAI model
chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=1)

# Session state setup
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

# LangChain memory integration
chat_message_store = StreamlitChatMessageHistory()
chat_memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    chat_memory=chat_message_store
)

# Prompt setup
chat_prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        SystemMessage(content="You are a helpful multilingual assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chat_chain = LLMChain(llm=chat_model, prompt=chat_prompt, memory=chat_memory)

# App Title
st.title("💬 Real-Time LLM Chatbot")

# Translation utility
def translate_response(text, target_lang):
    try:
        if target_lang == "en":
            return text
        max_chunk_size = 4500  # translation API limit
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_chunks = [GoogleTranslator(source='en', target=target_lang).translate(chunk) for chunk in chunks]
        return "".join(translated_chunks)
    except Exception as e:
        return f"⚠️ Translation error: {e}"

# Render conversation history
for turn in st.session_state.conversation_log:
    with st.chat_message("user"):
        st.markdown(turn["user"])

    with st.chat_message("assistant"):
        translated_text = translate_response(turn["gpt"], selected_lang_code)
        st.markdown(translated_text)

# Input box for user queries
user_query = st.chat_input("Type your message...")

if user_query:
    st.chat_message("user").markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 GPT is thinking..."):
            try:
                response = chat_chain.invoke({"content": user_query})
                assistant_response = response["text"]
            except Exception as e:
                assistant_response = f"⚠️ GPT failed: {e}"

            translated_output = translate_response(assistant_response, selected_lang_code)
            st.markdown(translated_output)

            # Log the full conversation
            st.session_state.conversation_log.append({
                "user": user_query,
                "gpt": assistant_response
            })
