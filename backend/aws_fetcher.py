import requests
from langchain.docstore.document import Document

def fetch_aws_docs(topic):
    # Simulated function for extracting AWS documentation text
    docs = [Document(page_content=f"Summary of {topic}", metadata={"url": "https://docs.aws.amazon.com"})]
    return docs