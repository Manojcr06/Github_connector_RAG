import requests
from langchain.docstore.document import Document

GITHUB_API_URL = "https://api.github.com"

def get_trending_repos(language="all"):
    # Fetch trending repos using GitHub's search API (filtered by stars for simplicity)
    response = requests.get(f"{GITHUB_API_URL}/search/repositories?q=stars:>5000+language:{language}&sort=stars")
    if response.status_code == 200:
        data = response.json()
        docs = [
            Document(
                page_content=f"{repo['name']}\n{repo['description'] or 'No description available.'}",
                metadata={"url": repo['html_url'], "stars": repo['stargazers_count']}
            ) for repo in data.get("items", [])
        ]
        return docs

def search_repos(query):
    response = requests.get(f"{GITHUB_API_URL}/search/repositories?q={query}+in:readme,description")
    if response.status_code == 200:
        data = response.json()
        docs = [
            Document(
                page_content=f"{repo['name']}\n{repo['description'] or 'No description available.'}",
                metadata={"url": repo['html_url'], "stars": repo['stargazers_count']}
            ) for repo in data.get("items", [])
        ]
        return docs
    
def fetch_github_data(query):
    response = requests.get(f"{GITHUB_API_URL}/search/repositories?q={query}+in:readme,description")
    if response.status_code == 200:
        data = response.json()
        docs = [
            Document(
                page_content=f"{repo['name']}\n{repo['description']}",
                metadata={"url": repo['html_url']}
            ) for repo in data.get("items", [])
        ]
        return docs
