import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader

load_dotenv()

import streamlit as st
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ["PINECONE_INDEX_NAME"] = st.secrets.get("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME", ""))

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Summarizer & Q&A")

# ── Session state init ──
for key, val in {
    "pdf_ready": False, "summary": "", "vectorstore": None,
    "chat_history": [], "answer": "", "retrieved_chunks": [],
    "retry_count": 0, "waiting_feedback": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ────────────────────────────────────────
# YOUR EXACT CODE BELOW — nothing changed
# ────────────────────────────────────────


class NodeData(TypedDict):
    pdf_path: str
    raw_text: str
    chunks: List[str]
    summary: str
    vectorstore: object
    user_input: str
    intent: str
    retrieved_chunks: List[str]
    answer: str
    answer_quality: str
    retry_count: int
    chat_history: List[dict]


def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )


def load_pdf(state: NodeData) -> dict:
    try:
        reader = PdfReader(state["pdf_path"])
        pages = [page.extract_text()
                 for page in reader.pages if page.extract_text()]
        raw_text = "\n".join(pages).strip()
        return {"raw_text": raw_text}          # ← fixed: return dict
    except Exception as e:
        return {"raw_text": f"Error: {str(e)}"}


def chunk_text(state: NodeData) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    chunks = splitter.split_text(state["raw_text"])
    return {"chunks": chunks}


def embed_store(state: NodeData) -> dict:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = PineconeVectorStore.from_texts(
        texts=state["chunks"],
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    return {"vectorstore": vectorstore}


def summarize(state: NodeData) -> dict:
    llm = get_llm()
    messages = [
        {"role": "system", "content": "Summarize the following document clearly and concisely."},
        {"role": "user", "content": "\n\n".join(state["chunks"][:20])}
    ]
    response = llm.invoke(messages)
    return {"summary": response.content, "retry_count": 0, "chat_history": []}


def query_process(state: NodeData) -> dict:
    query = state["user_input"]
    vectorstore = state["vectorstore"]
    results = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in results])
    llm = get_llm()
    messages = [
        {"role": "system", "content": "Answer based only on the context provided. If the answer is not in the context, say 'I could not find this in the document.'"},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
    ]
    response = llm.invoke(messages)
    answer = response.content
    history = state.get("chat_history", [])
    history.append({"question": query, "answer": answer})
    return {
        "answer": answer,
        "retrieved_chunks": [doc.page_content for doc in results],
        "chat_history": history
    }


def quality_check(state: NodeData) -> dict:
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        return {"answer_quality": "good", "retry_count": retry_count}
    query = state["user_input"]
    context = "\n\n".join(state.get("retrieved_chunks", []))
    llm = get_llm()
    messages = [
        {"role": "system",
            "content": f"The previous answer was not satisfactory. Please provide a more detailed and thorough answer. Previous answer: {state['answer']}. Answer based only on the context provided."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
    ]
    response = llm.invoke(messages)
    return {
        "answer": response.content,
        "answer_quality": "retry",
        "retry_count": retry_count + 1
    }


def router(state: NodeData) -> Literal["query_process", "end"]:
    if state["user_input"].strip().lower() == "exit":
        return "end"
    return "query_process"


def quality_router(state: NodeData) -> Literal["get_user_input", "quality_check"]:
    if state["answer_quality"] == "good":
        return "get_user_input"
    return "quality_check"

# ────────────────────────────────────────
# STREAMLIT REPLACES: input() and print()
# ────────────────────────────────────────


# Step 1 — Upload PDF (replaces: pdf_path = input(...))
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.pdf_ready:
    if st.button("Process PDF"):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        state = {"pdf_path": temp_path}
        with st.spinner("Reading PDF..."):
            state.update(load_pdf(state))
        with st.spinner("Chunking..."):
            state.update(chunk_text(state))
        with st.spinner("Embedding..."):
            state.update(embed_store(state))
        with st.spinner("Summarizing..."):
            state.update(summarize(state))
        os.remove(temp_path)
        st.session_state.vectorstore = state["vectorstore"]
        st.session_state.summary = state["summary"]
        st.session_state.pdf_ready = True
        st.rerun()

# Step 2 — Show summary (replaces: print(summary))
if st.session_state.pdf_ready:
    st.markdown("### 📋 Summary")
    st.info(st.session_state.summary)

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Step 3 — Quality check buttons (replaces: input("Is the answer good? yes/no"))
    if st.session_state.waiting_feedback:
        st.markdown("**Was this answer helpful?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.session_state.waiting_feedback = False
                st.session_state.retry_count = 0
                st.rerun()
        with col2:
            if st.button("👎 No — Retry"):
                with st.spinner("Retrying..."):
                    result = quality_check({
                        "user_input": st.session_state.chat_history[-1]["question"],
                        "answer": st.session_state.answer,
                        "retrieved_chunks": st.session_state.retrieved_chunks,
                        "retry_count": st.session_state.retry_count
                    })
                st.session_state.chat_history[-1]["answer"] = result["answer"]
                st.session_state.answer = result["answer"]
                st.session_state.retry_count = result["retry_count"]
                if result["retry_count"] >= 2:
                    st.session_state.waiting_feedback = False
                st.rerun()

    # Step 4 — Query input (replaces: user_input = input("You: "))
    if not st.session_state.waiting_feedback:
        query = st.chat_input("Ask a question about the document...")
        if query:
            if query.strip().lower() == "exit":
                st.success("👋 Goodbye! Upload a new PDF to start again.")
                for key in ["pdf_ready", "summary", "vectorstore", "chat_history",
                    "answer", "retrieved_chunks", "retry_count", "waiting_feedback"]:
                    del st.session_state[key]
                st.stop()
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = query_process({
                        "user_input": query,
                        "vectorstore": st.session_state.vectorstore,
                        "chat_history": st.session_state.chat_history
                    })
                st.write(result["answer"])
            st.session_state.answer = result["answer"]
            st.session_state.retrieved_chunks = result["retrieved_chunks"]
            st.session_state.chat_history = result["chat_history"]
            st.session_state.waiting_feedback = True
            st.rerun()

    if st.button("🔄 Upload New PDF"):
        for key in ["pdf_ready", "summary", "vectorstore", "chat_history",
                    "answer", "retrieved_chunks", "retry_count", "waiting_feedback"]:
            del st.session_state[key]
        st.rerun()
