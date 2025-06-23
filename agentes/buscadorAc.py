from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage

class SearchScholarAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="AcademicSearch",
                func=self.search_tool.run,
                description="Search for academic papers and research articles"
            )
        ]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are SearchScholar, an expert in academic research. Given a research topic, identify the most relevant papers published in the last 5 years. Prioritize high-citation papers from reliable sources. For each paper, provide: title, authors, year, and executive summary."""),
            ("user", "{input}"),
        ])
        self.agent = create_openai_functions_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools)
    
    def run(self, research_topic):
        response = self.agent_executor.invoke({
            "input": f"Find academic papers about: {research_topic}"
        })
        return response["output"]