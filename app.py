import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")

st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask questions based on Zyro Dynamics HR policy documents.")

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."


@st.cache_resource
def build_rag():
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    corpus_path = "data"

    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512
    )

    rag_prompt = ChatPromptTemplate.from_template("""
You are Zyro Dynamics HR Help Desk assistant.

Answer the employee question using ONLY the context provided below.

Rules:
- Use only the given context.
- Do not use outside knowledge.
- If the answer is not available in the context, say:
  "I could not find this information in the Zyro Dynamics HR policy documents."
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
""")

    oos_prompt = ChatPromptTemplate.from_template("""
You are a strict classifier.

Decide whether the question is related to Zyro Dynamics HR policies.

HR-related topics include:
leave, work from home, code of conduct, compensation, benefits, performance review,
probation, onboarding, separation, travel, expenses, IT policy, data security,
POSH, company profile, employee handbook.

Question:
{question}

Reply with only one word:
IN_SCOPE or OUT_OF_SCOPE
""")

    return retriever, llm, rag_prompt, oos_prompt


def format_docs(docs):
    formatted = []

    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown source"))
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"Source: {source}, Page: {page}\n{doc.page_content}")

    return "\n\n".join(formatted)


retriever, llm, rag_prompt, oos_prompt = build_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask an HR policy question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            guardrail_chain = oos_prompt | llm | StrOutputParser()
            decision = guardrail_chain.invoke({"question": question}).strip().upper()

            if "OUT_OF_SCOPE" in decision:
                answer = REFUSAL_MESSAGE
                sources = []
            else:
                docs = retriever.invoke(question)
                context = format_docs(docs)

                chain = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "question": question
                })

                sources = docs

            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for doc in sources[:5]:
                        source = os.path.basename(doc.metadata.get("source", "Unknown source"))
                        page = doc.metadata.get("page", "N/A")
                        st.write(f"- {source}, page {page}")

    st.session_state.messages.append({"role": "assistant", "content": answer})