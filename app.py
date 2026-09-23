import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(page_title="Smart Document Assistant")

st.title("📚 Smart Document Assistant")
st.write("Upload a PDF or TXT document and ask questions from it")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt"]
)


if uploaded_file:

    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("Document uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            if uploaded_file.name.lower().endswith(".pdf"):
                 loader = PyPDFLoader(file_path)

            elif uploaded_file.name.lower().endswith(".txt"):
                 loader = TextLoader(file_path, encoding="utf-8")

            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
          )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            
        st.success("Vector database created!")

        st.session_state["docs"] = docs

if "docs" in st.session_state:

    st.divider()

    st.subheader("📄 Document Summary")

    if st.button("Generate Document Summary"):

        with st.spinner("Generating document summary..."):

            docs = st.session_state["docs"]

            document_text = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            document_text = document_text[:30000]

            summary_llm = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                temperature=0
            )

            summary_prompt = f"""
You are a document summarization assistant.

Summarize the following document using ONLY the
information provided in the document.

Include:
1. Main topic of the document
2. Important concepts
3. Important sections or ideas
4. A short overall summary

Do not add information from outside the document.

Document:

{document_text}
"""

            summary_response = summary_llm.invoke(summary_prompt)

            if isinstance(summary_response.content, list):

                summary = "".join(
                    block.get("text", "")
                    for block in summary_response.content
                    if isinstance(block, dict)
                )

            else:
                summary = summary_response.content

            st.write(summary)



if os.path.exists("chroma_db"):

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)


        if isinstance(response.content, list):
            answer = "".join(
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict)
            )
        else:
            answer = response.content

        st.write("🤖 AI Answer")
        st.write(answer)

        st.write("### 📚 Sources")

        seen_pages = set()

        for doc in docs:
            page = doc.metadata.get("page")

            if page is not None:
                page_number = page + 1

                if page_number not in seen_pages:
                    st.write(f"- Page {page_number}")
                    seen_pages.add(page_number)
            else:
                st.write("- Page information unavailable")
