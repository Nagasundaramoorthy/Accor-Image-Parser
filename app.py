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

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_key = os.getenv('GEMINI_KEY')
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Hugging Face embeddings and LLM
embeddings = HuggingFaceEmbeddings()
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.9, "max_length": 500})

# Function to generate a concise image description
def gemini_response(img):
    response = gemini_model.generate_content([
        "Provide a concise and structured description of the image in 5 bullet points. Focus on key elements, colors, objects, and any notable details. Avoid unnecessary text.",
        img
    ])
    return response.text

# Function to generate a detailed image description
def gemini_response_desc(img):
    response = gemini_model.generate_content([
        "Describe the image in detail, covering all visible elements, their relationships, colors, textures, and any potential context or story the image might convey. Be thorough and engaging.",
        img
    ])
    return response.text

# Function to set up RAG (Retrieval-Augmented Generation)
def setup_rag(description):
    vector_store = FAISS.from_texts([description], embeddings)
    retriever = vector_store.as_retriever()

    prompt_template = """
    You are a knowledgeable and friendly assistant. Your task is to provide detailed, accurate, and engaging answers to the user's questions based on the context provided below.

    Context: {context}
    Question: {question}

    Answer in a conversational tone, ensuring clarity and depth. If the question cannot be answered from the context, politely inform the user.
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

# Function to generate a response using Gemini
def hybrid_response(question, context):
    response = gemini_model.generate_content([
        f"Context: {context}\n\nAnswer the following question in detail and clear and short: {question}"
    ])
    return response.text

# Streamlit app
st.title("Image Explanation App")
st.markdown("""
    Welcome to the Image Explanation App! Upload an image, and I'll describe it for you. 
    You can also ask me anything about the image, and I'll do my best to answer.
""")
st.write("### Upload an Image")
uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing image and generating description..."):
        try:
            # Generate image descriptions
            explanation = gemini_response(image)
            explanation_2 = gemini_response_desc(image)

            # Display the concise description
            st.write("### Image Description:")
            st.write(explanation)

            # Set up RAG for chatbot
            qa_chain = setup_rag(explanation_2)

            # Chatbot section
            st.write("### Chatbot:")
            # st.markdown("**Example questions you can ask:**")
            # st.markdown("- What is the main object in this image?")
            # st.markdown("- Can you describe the colors and textures?")
            # st.markdown("- What might be happening in this scene?")
            user_query = st.text_input("Ask anything about the image:")

            if user_query:
                with st.spinner("Generating response..."):
                    # Fetch relevant context using RAG
                    rag_response = qa_chain.run(user_query)
                    
                    # Generate a detailed response using Gemini
                    response = hybrid_response(user_query, rag_response)
                    st.session_state.chat_history.append((user_query, response))

            # Display chat history
            if st.session_state.chat_history:
                st.write("### Chat History:")
                for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
                    st.markdown(f"**Q{i}:** {question}")
                    st.markdown(f"**A{i}:** {answer}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please try uploading the image again or rephrasing your question.")
