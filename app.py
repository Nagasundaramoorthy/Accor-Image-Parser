import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

load_dotenv()

gemini_key = os.getenv('GEMINI_KEY')

genai.configure(api_key = gemini_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

dimension = 384
index = faiss.IndexFlatL2(dimension)
dish_names = []

def gemini_response(img):
    response = gemini_model.generate_content(
        [
            "Extract the dish names from the image. Return each dish name on a new line, No other text should be included in the response.",
            img
        ]
    )
    # print(response.text)
    return response.text.strip().split('\n')

st.title("Dish Availability Checker")
st.write("Upload an image containing dishes and query if a specific dish is available.")

upload_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])
if upload_file:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded Image",use_container_width =True)

    with st.spinner("Processing image..."):
        dishes = gemini_response(image)
        print(dishes)
        st.success("Dishes extracted!")
        st.write("Extracted dishes:",dishes)

        embeddings = embedding_model.encode(dishes)
        index.add(np.array(embeddings))
        dish_names.extend(dishes)

query =st.text_input("Ask about a dish (eg., 'Is Pizza available')")
if query:
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding),k=1)
    matched_dish = dish_names[indices[0][0]]

    if matched_dish.lower() in query.lower():
        st.success(f"yes, {matched_dish} is avaialble.")
    else:
        st.error("No, the dish is not available.")