from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PaperDigestorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._setup_chain()
    
    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are PaperDigestor, an academic synthesis expert. Summarize the provided research paper in 3 key points: 
             1) Main objective, 2) Methodology used, 3) Relevant conclusions. Keep it technical but accessible (max 200 words)."""),
            ("user", "{paper_content}"),
        ])
        return prompt | self.llm | StrOutputParser()
    
    def summarize(self, paper_content):
        return self.chain.invoke({"paper_content": paper_content})