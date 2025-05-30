import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

# ✅ 1. API 키를 secrets.toml에서 가져와 환경 변수로 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read())
    return file_path

def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    return doc

def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"]  # ✅ 명시적으로 키 전달
    )
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

############################### 2단계 : RAG 기능 ##########################

@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"]  # ✅
    )

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    retrieve_docs: List[Document] = retriever.invoke(user_question)

    chain = get_rag_chain()
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs

def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 상황
    - 교사의 권장반응
    - 생활지도 방법법
    - 법적근거
    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.environ["OPENAI_API_KEY"]  # ✅
    )

    return custom_rag_prompt | model | StrOutputParser()

############################### 3단계 : PDF 페이지 이미지 처리 ##########################

@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []

    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

############################### 4단계 : Streamlit UI ##########################

def main():
    st.set_page_config("학생생활지도 FAQ 챗봇", layout="wide")

    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.header("학생생활지도 FAQ 챗봇")

        pdf_doc = st.file_uploader("PDF Uploader", type="pdf")
        button = st.button("PDF 업로드하기")

        if pdf_doc and button:
            with st.spinner("PDF문서 저장중"):
                pdf_path = save_uploadedfile(pdf_doc)
                pdf_document = pdf_to_documents(pdf_path)
                smaller_documents = chunk_documents(pdf_document)
                save_to_vector_store(smaller_documents)

            with st.spinner("PDF 페이지를 이미지로 변환중"):
                images = convert_pdf_to_images(pdf_path)
                st.session_state.images = images

        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요", placeholder="수업시간에 돌아다니는 학생이 있는 경우")

        if user_question:
            response, context = process_question(user_question)
            st.write(response)

            for i, document in enumerate(context):
                with st.expander("관련 문서"):
                    st.write(document.page_content)
                    file_path = document.metadata.get('source', '')
                    page_number = document.metadata.get('page', 0) + 1
                    button_key = f"link_{file_path}_{page_number}_{i}"
                    reference_button = st.button(f"🔎 {os.path.basename(file_path)} pg.{page_number}", key=button_key)
                    if reference_button:
                        st.session_state.page_number = str(page_number)

    with right_column:
        page_number = st.session_state.get('page_number')
        if page_number:
            page_number = int(page_number)
            image_folder = "PDF_이미지"
            images = sorted(os.listdir(image_folder), key=natural_sort_key)
            image_paths = [os.path.join(image_folder, img) for img in images]
            display_pdf_page(image_paths[page_number - 1], page_number)

if __name__ == "__main__":
    main()
    