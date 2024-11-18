import streamlit as st
import os
import numpy as np
from backend.vector_store import create_and_store_vector_store, load_vector_store
from backend.github_fetcher import get_trending_repos, search_repos, fetch_github_data
# from backend.aws_fetcher import fetch_aws_docs
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
from transformers import pipeline
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer


# Load pre-existing vector stores
github_vector_store_path = "data/github_docs.pkl"
# github_vector_store = load_vector_store(github_vector_store_path)
faiss_index, github_docs = load_vector_store(github_vector_store_path)
aws_vector_store_path = "data/aws_docs.pkl"
# aws_vector_store = load_vector_store(aws_vector_store_path)

# Initialize OpenAI LLM with the API key
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("OpenAI API key is not set. Please configure it.")
# llm = OpenAI(openai_api_key=api_key)

qa_model = pipeline('text2text-generation', model='google/flan-t5-base', max_length = 512)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_k_docs(query, k=20):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    _, indices = faiss_index.search(np.array([query_embedding]), k)
    return [github_docs[idx] for idx in indices[0]]

def display_repo_card(repo_doc, task_type):
    # Check if page content exists
    if not hasattr(repo_doc, 'page_content') or not repo_doc.page_content:
        st.warning("No content available for this repository.")
        return

    # Extract and split the page content for the repository's name and description
    content = repo_doc.page_content.split('\n')
    repo_name = content[0] if len(content) > 0 else "Unknown Repository"
    repo_description = content[1] if len(content) > 1 else "No description available."

    # Extract ID, stars, and URL from metadata, with fallback values if keys are missing
    repo_id = repo_doc.id if repo_doc.id else "N/A"
    stars = repo_doc.metadata.get('stars', 'N/A')
    repo_url = repo_doc.metadata.get('url', '#')

    # Customize display based on the task type
    if task_type == "Trending Repos":
        card_type = "Trending Repository"
    elif task_type == "Search Repos":
        card_type = "Search Result"
    elif task_type == "Fetch GitHub Data":
        card_type = "Fetched Repository Data"
    else:
        card_type = "Repository"

    # Display card with extracted information including ID
    st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #f9f9f9;">
            <h4>{repo_name} <span style="color: #888; font-size: 0.9em;">({card_type})</span></h4>
            <p><strong>ID:</strong> {repo_id}</p>
            <p>{repo_description}</p>
            <p><strong>Stars:</strong> {stars}</p>
            <p><a href="{repo_url}" target="_blank">Visit the Repository</a></p>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.title("Centralized Github RAG Application")
    toggle_scrape = st.checkbox("Enable Live Data Fetching")

    option = st.selectbox("Choose your data source:", ("GitHub", "AWS Documentation"))

    if option == "GitHub":
        subtask = st.radio("Choose a subtask:", ("Trending Repos", "Search Repos", "Fetch GitHub Data"))

        if toggle_scrape:
            if subtask == "Trending Repos":
                st.subheader("Fetching Trending GitHub Repositories...")
                docs = get_trending_repos("all")
                if st.button("Store in Vector DB"):
                    create_and_store_vector_store(docs, github_vector_store_path)
                    st.success("Trending data stored in Vector DB.")
                for doc in docs:
                    display_repo_card(doc, task_type="Trending Repos") 

            elif subtask == "Search Repos":
                search_query = st.text_input("Enter a technology or topic:")
                if st.button("Search and Display"):
                    search_docs = search_repos(search_query)
                    create_and_store_vector_store(search_docs, github_vector_store_path)
                    st.success("The Repos Searched were stored in Vector DB.")
                    for doc in search_docs:
                        display_repo_card(doc, task_type="Search Repos")

            elif subtask == "Fetch GitHub Data":
                search_query = st.text_input("Enter a topic to fetch data:")
                if st.button("Fetch"):
                    fetch_docs = fetch_github_data(search_query)
                    create_and_store_vector_store(fetch_docs, github_vector_store_path)
                    st.success("The Fetched Data were stored in Vector DB.")
                    for doc in fetch_docs:
                           display_repo_card(doc, task_type="Fetch Github Data") 

        else:
            # st.write("Using pre-stored Vector DB data.")
            # retriever = github_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            # query = st.text_input("Ask about GitHub data:")
            # if st.button("Query"):
            #     qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            #     answer = qa_chain.run(query)
            #     st.write("Answer:", answer)
            st.write("Using pre-stored Vector DB data.")
            query = st.text_input("Ask about GitHub data:")
            if st.button("Query"):
                top_docs = get_top_k_docs(query, k=20)
                context = " ".join([doc.page_content for doc in top_docs])
                response = qa_model(f"Question: {query} Context: {context}")
                st.write("Answer:", response[0]['generated_text'])

    elif option == "AWS Documentation":
            topic = st.text_input("Enter AWS documentation topic:")
            # if st.button("Fetch Data"):
            #     docs = fetch_aws_docs(topic)
            #     st.write("Fetched data displayed directly.")
            #     for doc in docs:
            #         st.markdown(f"{doc.page_content}\n[Link]({doc.metadata['url']})")

    # else:
    #     st.write("Using pre-stored Vector DB.")
    #     llm = OpenAI()
    #     if option == "GitHub":
    #         retriever = github_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    #     elif option == "AWS Documentation":
    #         retriever = aws_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    #     query = st.text_input("Ask a question:")
    #     if st.button("Query"):
    #         qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    #         answer = qa_chain.run(query)
    #         st.write("Answer:", answer)

if __name__ == "__main__":
    main()