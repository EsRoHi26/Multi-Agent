from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class CiteMasterAgent:
    def __init__(self, llm, vector_db_path):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = self._load_vector_db(vector_db_path)
        self.chain = self._setup_chain()
    
    def _load_vector_db(self, path):
        return FAISS.load_local(path, self.embeddings)
    
    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are CiteMaster, an academic librarian. Given this text {document} and our reference database:
             1) Verify all citations are correct, 2) Suggest APA/MLA formats, 3) For missing citations, recommend relevant papers from our database that support the arguments."""),
            ("user", "Process document: {document}"),
        ])
        
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        return (
            {"context": retriever, "document": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def process_document(self, document):
        return self.chain.invoke({"document": document})
    
    def add_to_db(self, papers):
        # Método para añadir nuevos papers a la base vectorial
        texts = [f"{p['title']}\n{p['authors']}\n{p['abstract']}" for p in papers]
        metadatas = [{"title": p["title"], "year": p["year"]} for p in papers]
        self.vector_db.add_texts(texts, metadatas)
        self.vector_db.save_local("database/papers_faiss_index")