import os
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader
from IPython.display import Image, display

load_dotenv()

# ----------------------------------------
# STATE
# ----------------------------------------


class NodeData(TypedDict):

    # Document
    pdf_path: str
    raw_text: str
    chunks: List[str]
    summary: str
    vectorstore: PineconeVectorStore

    # Q&A
    user_input: str
    intent: str           # "question" or "exit"
    retrieved_chunks: List[str]
    answer: str
    answer_quality: str   # "good" or "retry"
    retry_count: int

    # History
    chat_history: List[dict]

# ----------------------------------------
# HELPER — LLM
# ----------------------------------------


def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="qwen/qwen3.6-27b",
        temperature=0.0
    )


# --------------------------------------------------------
#  NODES — all accept state: NodeData, return dict
# ---------------------------------------------------------
def load_pdf(state: NodeData) -> dict:
    try:
        reader = PdfReader(state["pdf_path"])
        pages = [page.extract_text()
                 for page in reader.pages if page.extract_text()]
        return "\n".join(pages).strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def chunk_text(state: NodeData) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
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
        {"role": "user", "content": "\n\n".join(state["chunks"])}
    ]
    response = llm.invoke(messages)
    summary = response.content
    print(f"\n📄 DOCUMENT SUMMARY:\n{summary}\n")
    print("─" * 60)
    return {"summary": summary, "retry_count": 0, "chat_history": []}


def get_user_input(state: NodeData) -> dict:
    print("\nAsk a question about the document (or type 'exit' to quit):")
    user_input = input("You: ").strip()
    return {"user_input": user_input}


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

    print(f"\nAnswer:\n{answer}\n")
    return {
        "answer": answer,
        "retrieved_chunks": [doc.page_content for doc in results],
        "chat_history": history
    }


def quality_check(state: NodeData) -> dict:
    feedback = input("Is the answer good? (yes/no): ").strip().lower()
    if feedback == "yes":

        return {"answer_quality": "good"}

    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        print("Maximum retries reached. Moving on.")
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
    better_answer = response.content
    print(f"\nImproved Answer:\n{better_answer}\n")

    return {
        "answer": better_answer,
        "answer_quality": "retry",
        "retry_count": retry_count + 1
    }

# ---------------------------------------------
#   CONDITIONAL EDGE
# ---------------------------------------------


def router(state: NodeData) -> Literal["query_process", "end"]:
    if state["user_input"].strip().lower() == "exit":
        print("👋 Goodbye!")
        return "end"
    return "query_process"


def quality_router(state: NodeData) -> Literal["get_user_input", "quality_check"]:
    if state["answer_quality"] == "good":
        return "get_user_input"
    return "quality_check"

# ---------------------------------------------
#   BUILD GRAPH
# ---------------------------------------------


def build_graph():
    builder = StateGraph(NodeData)
    builder.add_node("load_pdf", load_pdf)
    builder.add_node("chunk_text", chunk_text)
    builder.add_node("embed_store", embed_store)
    builder.add_node("summarize", summarize)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("query_process", query_process)
    builder.add_node("quality_check", quality_check)

    builder.add_edge(START, "load_pdf")
    builder.add_edge("load_pdf", "chunk_text")
    builder.add_edge("chunk_text", "embed_store")
    builder.add_edge("embed_store", "summarize")
    builder.add_edge("summarize", "get_user_input")

    builder.add_conditional_edges("get_user_input", router, {
        "query_process": "query_process",
        "end": END
    })
    builder.add_edge("query_process", "quality_check")
    builder.add_conditional_edges("quality_check", quality_router, {
        "get_user_input": "get_user_input",
        "quality_check": "quality_check"
    })

    graph = builder.compile()
    return graph


# ---------------------------------------------
#   MAIN
# ---------------------------------------------
if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
    else:
        graph = build_graph()
        initial_state = {
            "pdf_path": pdf_path,
            "raw_text": "",
            "chunks": [],
            "summary": "",
            "vectorstore": None,
            "user_input": "",
            "intent": "",
            "retrieved_chunks": [],
            "answer": "",
            "answer_quality": "good",
            "retry_count": 0,
            "chat_history": []
        }
        graph.invoke(initial_state)
