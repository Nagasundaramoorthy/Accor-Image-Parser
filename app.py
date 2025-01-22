import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

load_dotenv()

gemini_key = os.getenv('GEMINI_KEY')
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

embeddings = HuggingFaceEmbeddings()
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.9, "max_length": 500})

def gemini_response(img):
    response = gemini_model.generate_content(['Describe the image in points. Max it should not exceed 5 points. No other additional text should be in the response.', img])
    return response.text

def gemini_response_desc(img):
    response = gemini_model.generate_content(['Describe in detail about the image.', img])
    return response.text

def setup_rag(description):
    vector_store = FAISS.from_texts([description], embeddings)
    retriever = vector_store.as_retriever()

    prompt_template = """
    You are a knowledgeable and helpful assistant. Based on the context provided below, answer the user's question in a detailed and conversational manner.

    Context: {context}
    Question: {question}
    Answer in detail:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

st.title("Image Explanation App")
st.write("Upload an Image.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        explanation = gemini_response(image)
        explanation_2 = gemini_response_desc(image)
        st.write("### Image Description:")
        st.write(explanation)

        qa_chain = setup_rag(explanation_2)
        st.write("### Chatbot:")
        user_query = st.text_input("Ask anything about the image:")
        if user_query:
            response = qa_chain.run(user_query)
            st.session_state.chat_history.append((user_query, response))

        if st.session_state.chat_history:
            st.write("### Chat History:")
            for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**Q{i}:** {question}")
                st.markdown(f"**A{i}:** {answer}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
