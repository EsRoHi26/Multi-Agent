from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ResearchWriterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._setup_chain()
    
    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are ResearchWriter, a scientific writing expert. Based on these paper summaries {summaries} and research topic {topic}, write a coherent literature review section that:
             1) Contextualizes the topic, 2) Compares methodologies, 3) Identifies research gaps. Use appropriate citations and maintain formal academic tone."""),
            ("user", "Write literature review for: {topic}"),
        ])
        return (
            {"summaries": RunnablePassthrough(), "topic": RunnablePassthrough()} 
            | prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def write_review(self, summaries, topic):
        return self.chain.invoke({"summaries": summaries, "topic": topic})