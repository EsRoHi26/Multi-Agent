from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from agentes.buscadorAc import SearchScholarAgent
from agentes.resumidor import PaperDigestorAgent
from agentes.redactor import ResearchWriterAgent
from agentes.agRef import CiteMasterAgent
from langchain_openai import ChatOpenAI
import operator

# Definir el estado del grafo
class ResearchState(TypedDict):
    research_topic: str
    found_papers: List[str]
    paper_summaries: List[str]
    literature_review: str
    final_output: str

# Configurar agentes
llm = ChatOpenAI(model="gpt-4-1106-preview")
search_agent = SearchScholarAgent(llm)
digestor_agent = PaperDigestorAgent(llm)
writer_agent = ResearchWriterAgent(llm)
cite_agent = CiteMasterAgent(llm, "database/papers_faiss_index")

# Definir nodos del grafo
def search_papers(state: ResearchState):
    papers = search_agent.run(state["research_topic"])
    return {"found_papers": papers}

def summarize_papers(state: ResearchState):
    summaries = [digestor_agent.summarize(paper) for paper in state["found_papers"]]
    return {"paper_summaries": summaries}

def write_review(state: ResearchState):
    review = writer_agent.write_review(state["paper_summaries"], state["research_topic"])
    return {"literature_review": review}

def add_references(state: ResearchState):
    final_output = cite_agent.process_document(state["literature_review"])
    return {"final_output": final_output}

# Construir el grafo
workflow = StateGraph(ResearchState)
workflow.add_node("search", search_papers)
workflow.add_node("summarize", summarize_papers)
workflow.add_node("write", write_review)
workflow.add_node("cite", add_references)

# Definir bordes
workflow.set_entry_point("search")
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", "write")
workflow.add_edge("write", "cite")
workflow.add_edge("cite", END)

# Compilar el grafo
research_graph = workflow.compile()

# Función principal
def run_research_assistant(topic: str):
    results = research_graph.invoke({"research_topic": topic})
    return results["final_output"]

if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    result = run_research_assistant(topic)
    print("\n=== Research Assistance Result ===")
    print(result)