# 🤖 Real-Time LLM Chatbot with Translation (GPT + Streamlit)

A multilingual AI chatbot built using OpenAI GPT-4o, LangChain, and Streamlit, capable of translating responses into over 50+ languages in real time. Supports conversation memory, chat history, and dynamic language switching.

---

## 🌟 Features

- 🧠 Powered by OpenAI's GPT (gpt-4o-mini)
- 🌐 Multilingual support with 50+ languages including **Telugu**, **Hindi**, **Spanish**, **French**, etc.
- 💬 Real-time chat with GPT using Streamlit UI
- 📝 Chat history sidebar with scrollable view
- ♻️ Memory-enabled conversation via LangChain
- ⏳ Loading indicator with "GPT is thinking..." feedback
- ✅ Error handling for both GPT and translation API
- ✨ Clean UI with dynamic translation per question

---

## 📸 Demo

<img width="575" height="305" alt="image" src="https://github.com/user-attachments/assets/691a9df9-8d3a-4760-b1ea-5bd028f5f8ec" />

<img width="592" height="314" alt="image" src="https://github.com/user-attachments/assets/d12292e3-c6e1-4d8d-abc2-fda7977bf499" />

---

## 🧰 Tech Stack

- **Frontend/UI**: Streamlit
- **LLM**: OpenAI GPT-4o-mini via `langchain-openai`
- **Translation**: `deep-translator` (Google Translate)
- **Memory & Prompting**: LangChain (`LLMChain`, `ConversationBufferMemory`)
- **Environment**: Python, .env, virtualenv

---

