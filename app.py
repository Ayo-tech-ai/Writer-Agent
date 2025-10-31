# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + DUCKDUCKGO + CREWAI)
# =====================================================================

import os
import streamlit as st
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM, tools
from duckduckgo_search import DDGS
import time

# =====================================================================
# ⚙️ APP CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Groq Agentic Researcher", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # API Key input with option to use environment variable
    groq_api_key = st.text_input(
        "Enter your Groq API Key", 
        type="password",
        placeholder="sk-... or leave blank to use GROQ_API_KEY env var"
    )
    
    # Model selection
    model_options = {
        "Llama 3 70B": "llama3-70b-8192",
        "Llama 3 8B": "llama3-8b-8192", 
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma 7B": "gemma-7b-it"
    }
    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
    with col2:
        max_results = st.slider("Max Results", 1, 10, 5)
    
    st.markdown("---")
    st.info("💡 This app uses **DuckDuckGo Search** — no API key required!")
    
    # Additional info
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your Groq API key (or set GROQ_API_KEY environment variable)
        2. Type your research topic
        3. Click 'Run Research'
        4. Wait for the agents to complete their work
        
        **Agents:**
        - 🔍 **Researcher**: Searches the web for information
        - ✍️ **Writer**: Summarizes findings into a coherent report
        """)

# =====================================================================
# 🧠 INITIALIZE GROQ LLM
# =====================================================================

def initialize_llm(api_key, model_name, temperature):
    """Initialize Groq LLM with error handling."""
    try:
        # Use provided key or environment variable
        final_api_key = api_key.strip() if api_key.strip() else os.getenv("GROQ_API_KEY")
        
        if not final_api_key:
            st.error("❌ No Groq API key provided. Please enter your API key or set GROQ_API_KEY environment variable.")
            return None
            
        return LLM(
            model=model_name,
            api_key=final_api_key,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {str(e)}")
        return None



# =====================================================================
# 👥 DEFINE # =====================================================================
# 🌐 DUCKDUCKGO SEARCH TOOL (UPDATED)
# =====================================================================

@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        results = []
        with DDGS() as ddg:
            for result in ddg.text(query, max_results=max_results):
                title = result.get("title", "No title")
                href = result.get("href", "No URL")
                body = result.get("body", "No description")
                results.append(f"**{title}**\n{href}\n{body}\n")
        
        if not results:
            return "No results found for the given query."
            
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Error performing search: {str(e)}"


def create_agents(llm, search_tool):
    """Create research and writer agents."""
    research_agent = Agent(
        role="Senior Web Research Analyst",
        goal="Gather the most relevant, accurate, and recent online information from trustworthy sources.",
        backstory="You are an expert online researcher with years of experience in finding reliable information across various domains. You excel at distinguishing credible sources from unreliable ones.",
        llm=llm,
        tools=[search_tool],
        allow_delegation=False,
        verbose=True
    )

    writer_agent = Agent(
        role="Senior Technical Writer",
        goal="Transform research findings into clear, engaging, and well-structured summaries that are easy to understand.",
        backstory="You are a professional writer specializing in technical content. You have a talent for making complex information accessible and engaging for diverse audiences.",
        llm=llm,
        allow_delegation=False,
        verbose=True
    )
    
    return research_agent, writer_agent

# =====================================================================
# 🧾 DEFINE TASKS
# =====================================================================

def create_tasks(research_agent, writer_agent, query, max_results):
    """Create research and writing tasks."""
    research_task = Task(
        description=f"""
        Conduct comprehensive online research on: "{query}"
        
        Requirements:
        - Search for the most recent and relevant information
        - Focus on credible sources and authoritative websites
        - Gather diverse perspectives on the topic
        - Extract key facts, statistics, and insights
        - Return {max_results} most valuable findings
        """,
        expected_output=f"A comprehensive bullet-point summary of the {max_results} most relevant and credible insights found during web research.",
        agent=research_agent
    )

    summary_task = Task(
        description="""
        Analyze the research findings and create a professional summary that:
        - Highlights the most important information
        - Presents information in a logical flow
        - Uses clear, concise language
        - Includes key takeaways
        - Is engaging and easy to read
        """,
        expected_output="A well-structured 3-4 paragraph summary with clear headings, key insights, and actionable conclusions.",
        agent=writer_agent
    )
    
    return research_task, summary_task

# =====================================================================
# ⚙️ MAIN EXECUTION
# =====================================================================

def main():
    """Main application logic."""
    
    # User input
    query = st.text_area(
        "🔎 Enter your research topic:", 
        placeholder="e.g. Emerging AI applications in African agriculture, Latest developments in quantum computing, Sustainable energy trends 2024...",
        height=100
    )
    
    # Initialize LLM
    llm = initialize_llm(groq_api_key, model_options[selected_model], temperature)
    
    if not llm:
        return
    
    # Create agents and tools
    research_agent, writer_agent = create_agents(llm, duckduckgo_tool)
    
    # Execute research
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Run Research", use_container_width=True, type="primary"):
            if not query.strip():
                st.warning("⚠️ Please enter a research topic.")
            else:
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Update progress
                    status_text.text("🔄 Initializing research crew...")
                    progress_bar.progress(20)
                    
                    # Create tasks
                    research_task, summary_task = create_tasks(
                        research_agent, writer_agent, query, max_results
                    )
                    
                    # Create crew
                    crew = Crew(
                        agents=[research_agent, writer_agent],
                        tasks=[research_task, summary_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    # Execute
                    status_text.text("🔍 Researching topic...")
                    progress_bar.progress(50)
                    
                    status_text.text("📝 Writing summary...")
                    progress_bar.progress(80)
                    
                    result = crew.kickoff()
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("✅ Research complete!")
                    
                    # Display results
                    st.success("✅ Research complete!")
                    
                    # Results in expandable sections
                    with st.expander("📋 Executive Summary", expanded=True):
                        st.write(result)
                    
                    # Additional information
                    with st.expander("🔧 Research Details"):
                        st.write(f"**Model:** {selected_model}")
                        st.write(f"**Temperature:** {temperature}")
                        st.write(f"**Max Results:** {max_results}")
                        st.write(f"**Query:** {query}")
                        
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("")

    # Display usage tips
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        - **Be specific** in your research topic for more relevant results
        - **Adjust temperature**: Lower (0.1-0.3) for factual topics, higher (0.7-1.0) for creative topics
        - **Use more results** for comprehensive research on complex topics
        - **Try different models** if you're not satisfied with the results
        """)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + CrewAI + DuckDuckGo Search | " +
          "Note: Research quality depends on search results and model capabilities")

if __name__ == "__main__":
    main()
