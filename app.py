import streamlit as st
import numpy as np
import pandas as pd
import wikipediaapi
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from scipy.spatial.distance import cosine as cosine_similarity

# Specific model being used
MODEL_NAME = 'textembedding-gecko@003'

# Function to get embeddings for texts
def embed_text(texts: list, model_name: str = MODEL_NAME) -> list:
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        inputs = [TextEmbeddingInput(text) for text in texts]
        embeddings = model.get_embeddings(inputs)
        return [np.array(embedding.values) for embedding in embeddings]
    except Exception as e:
        raise RuntimeError(f"Error embedding texts: {e}")

# Function to calculate cosine similarity between two vectors
def cosine_similarity_percentage(vec1: np.ndarray, vec2: np.ndarray) -> float:
    similarity = 1 - cosine_similarity(vec1, vec2)
    return similarity * 100

# Function to get Wikipedia content
def fetch_wikipedia_page(title):
    user_agent = 'WikiParagraphSearchApp/1.0 (saksath1016@gmail.com)'
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=user_agent
    )
    page = wiki_wiki.page(title)
    if not page.exists():
        raise ValueError("Page does not exist.")
    return page.text

# Function to save paragraphs to CSV
def save_paragraphs_to_csv(content, filename):
    paragraphs = content.split('\n\n')
    df = pd.DataFrame(paragraphs, columns=['paragraph'])
    df.to_csv(filename, index=False)

# Function to find the most relevant paragraph
def find_most_relevant_paragraph(question, paragraphs, embeddings):
    question_embedding = embed_text([question])[0]
    similarities = [cosine_similarity_percentage(question_embedding, paragraph_embedding) for paragraph_embedding in embeddings]
    most_relevant_idx = np.argmax(similarities)
    return paragraphs[most_relevant_idx]

# Streamlit application
def main():
    st.title("Wikipedia Paragraph Search")

    wikipedia_page = st.text_input("Enter Wikipedia Page Title:")

    if st.button("Fetch and Process Page"):
        try:
            content = fetch_wikipedia_page(wikipedia_page)
            save_paragraphs_to_csv(content, "wikipedia_page.csv")
            st.success("Page content saved to wikipedia_page.csv")
        except ValueError as e:
            st.error(f"Error: {e}")

    if st.button("Load Data and Embeddings"):
        df = pd.read_csv("wikipedia_page.csv")
        paragraphs = df['paragraph'].tolist()
        embeddings = embed_text(paragraphs)
        st.session_state['paragraphs'] = paragraphs
        st.session_state['embeddings'] = embeddings
        st.success("Data and embeddings loaded successfully")

    question = st.text_input("Ask a question:")

    if question and 'paragraphs' in st.session_state and 'embeddings' in st.session_state:
        paragraphs = st.session_state['paragraphs']
        embeddings = st.session_state['embeddings']
        most_relevant_paragraph = find_most_relevant_paragraph(question, paragraphs, embeddings)
        st.write("Most Relevant Paragraph:")
        st.write(most_relevant_paragraph)

if __name__ == '__main__':
    main()