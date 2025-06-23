from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import json

class VectorDatabase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.db = None
    
    def initialize_db(self, papers_json_path="data/sample_papers.json"):
        with open(papers_json_path, "r") as f:
            papers = json.load(f)
        
        texts = [f"{p['title']}\n{p['authors']}\n{p['abstract']}" for p in papers]
        metadatas = [{"title": p["title"], "year": p["year"]} for p in papers]
        
        self.db = FAISS.from_texts(texts, self.embeddings, metadatas)
        self.db.save_local("database/papers_faiss_index")
    
    def load_db(self, path="database/papers_faiss_index"):
        self.db = FAISS.load_local(path, self.embeddings)
        return self.db