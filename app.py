# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + DUCKDUCKGO + CREWAI)
# =====================================================================

import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.llm import GroqLLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# =====================================================================
# ⚙️ APP CONFIGURATION
# =====================================================================

st.set_page_config(page_title="Groq Agentic Researcher", page_icon="🌍", layout="wide")

st.title("🌍 Groq-Powered Agentic Researcher")
st.markdown(
    """
    🤖 **Powered by Groq + CrewAI + DuckDuckGo Search**  
    _Perform intelligent web research and receive concise summaries._
    """
)

# =====================================================================
# 🔑 API KEY INPUT
# =====================================================================

with st.sidebar:
    st.header("🔐 Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7)
    st.markdown("---")
    st.info("💡 This app uses **DuckDuckGo Search** — no API key required!")

if not groq_api_key and not os.getenv("GROQ_API_KEY"):
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# =====================================================================
# 🧠 STEP 1 — INITIALIZE GROQ LLM
# =====================================================================

groq_llm = GroqLLM(
    model="llama3-70b-8192",
    api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
    temperature=temperature
)

# =====================================================================
# 🌐 STEP 2 — DEFINE DUCKDUCKGO SEARCH TOOL (UPDATED)
# =====================================================================

@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        results = []
        with DDGS() as ddg:
            for result in ddg.text(query, max_results=max_results):
                title = result.get("title", "")
                href = result.get("href", "")
                body = result.get("body", "")
                results.append(f"**{title}**\n{href}\n{body}\n")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Initialize the tool
duckduckgo_tool = duckduckgo_search

# =====================================================================
# 👥 STEP 3 — DEFINE AGENTS
# =====================================================================

research_agent = Agent(
    role="Web Research Analyst",
    goal="Gather the most relevant and accurate online information.",
    backstory="You are a skilled online researcher who uses DuckDuckGo to find trustworthy and recent data.",
    llm=groq_llm,
    tools=[duckduckgo_tool],
    allow_delegation=False,
    verbose=True
)

writer_agent = Agent(
    role="Insight Writer",
    goal="Summarize research findings into well-structured insights.",
    backstory="You are an expert writer who turns technical data into engaging, clear summaries.",
    llm=groq_llm,
    allow_delegation=False,
    verbose=True
)

# =====================================================================
# 🧾 STEP 4 — DEFINE TASKS
# =====================================================================

research_task = Task(
    description="Conduct an online search on the user's topic and collect key findings.",
    expected_output="A bullet-point summary of the 5–10 most relevant insights found on the web.",
    agent=research_agent
)

summary_task = Task(
    description="Read the research findings and create a clear, concise professional summary.",
    expected_output="A 2–3 paragraph summary of the findings in natural language.",
    agent=writer_agent
)

# =====================================================================
# ⚙️ STEP 5 — USER INPUT + EXECUTION
# =====================================================================

query = st.text_area("🔎 Enter your research topic:", placeholder="e.g. Emerging AI applications in African agriculture")

if st.button("🚀 Run Research", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a topic to research.")
    else:
        with st.spinner("🤖 Agents are researching and summarizing... please wait."):
            crew = Crew(
                agents=[research_agent, writer_agent],
                tasks=[research_task, summary_task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff()
        
        st.success("✅ Research complete!")
        st.subheader("🧩 Summary:")
        st.write(result)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + CrewAI + DuckDuckGo Search")
