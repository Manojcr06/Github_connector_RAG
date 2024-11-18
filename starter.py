from backend.vector_store import create_and_store_vector_store
from backend.github_fetcher import fetch_github_data, get_trending_repos, search_repos
# from backend.aws_fetcher import fetch_aws_docs

# Fetch initial data and create vector stores
github_docs = fetch_github_data("LLM")
github_trends = get_trending_repos()
github_search = search_repos("RAG")
# aws_docs = fetch_aws_docs("How to use AWS EC2")

create_and_store_vector_store(github_trends, "data/github_docs.pkl")
create_and_store_vector_store(github_search, "data/github_docs.pkl")
create_and_store_vector_store(github_docs, "data/github_docs.pkl")
# create_and_store_vector_store(aws_docs, "data/aws_docs.pkl")