# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + DUCKDUCKGO + CREWAI)
# =====================================================================

import os
import streamlit as st
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process
from duckduckgo_search import DDGS
import requests
import json

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
        placeholder="gsk_... or leave blank to use GROQ_API_KEY env var"
    )
    
    # Model selection - Updated for latest Groq models
    model_options = {
        "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
        "Llama 3.1 8B Instant": "llama-3.1-8b-instant", 
        "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma2 9B": "gemma2-9b-it"
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
# 🌐 DUCKDUCKGO SEARCH TOOL
# =====================================================================

@tool
def duckduckgo_search_tool(query: str, max_results: int = 5) -> str:
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

# =====================================================================
# 🧠 CUSTOM GROQ LLM INTEGRATION
# =====================================================================

class GroqLLM:
    """Custom Groq LLM wrapper for CrewAI"""
    
    def __init__(self, api_key, model, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def call(self, messages, **kwargs):
        """Make API call to Groq"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4000,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Request failed: {str(e)}")
            return None
        except KeyError as e:
            st.error(f"❌ Unexpected API response format: {str(e)}")
            return None

# =====================================================================
# 👥 DEFINE AGENTS (UPDATED FOR CUSTOM GROQ INTEGRATION)
# =====================================================================

def create_agents(groq_llm, search_tool):
    """Create research and writer agents with custom Groq LLM."""
    
    # Custom agent class to work with our Groq LLM
    class GroqAgent:
        def __init__(self, role, goal, backstory, tools=None):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.tools = tools or []
            self.llm = groq_llm
            
        def execute_task(self, task_description):
            """Execute a task using the Groq LLM"""
            messages = [
                {"role": "system", "content": f"""You are a {self.role}. {self.backstory}
                 
                 Your goal: {self.goal}
                 
                 Available tools: {[tool.__name__ for tool in self.tools] if self.tools else 'None'}
                 
                 Current task: {task_description}"""},
                {"role": "user", "content": task_description}
            ]
            
            return self.llm.call(messages)

    research_agent = GroqAgent(
        role="Senior Web Research Analyst",
        goal="Gather the most relevant, accurate, and recent online information from trustworthy sources.",
        backstory="You are an expert online researcher with years of experience in finding reliable information across various domains. You excel at distinguishing credible sources from unreliable ones.",
        tools=[search_tool]
    )

    writer_agent = GroqAgent(
        role="Senior Technical Writer",
        goal="Transform research findings into clear, engaging, and well-structured summaries that are easy to understand.",
        backstory="You are a professional writer specializing in technical content. You have a talent for making complex information accessible and engaging for diverse audiences.",
        tools=[]
    )
    
    return research_agent, writer_agent

# =====================================================================
# 🧾 SIMPLIFIED CREW EXECUTION
# =====================================================================

def execute_research(query, research_agent, writer_agent, max_results):
    """Execute the research workflow"""
    
    # Step 1: Research
    research_prompt = f"""
    Conduct comprehensive online research on: "{query}"
    
    Requirements:
    - Search for the most recent and relevant information
    - Focus on credible sources and authoritative websites
    - Gather diverse perspectives on the topic
    - Extract key facts, statistics, and insights
    - Return {max_results} most valuable findings
    
    Use the search tool to find information and provide a comprehensive bullet-point summary.
    """
    
    st.info("🔍 Researching topic...")
    research_results = research_agent.execute_task(research_prompt)
    
    if not research_results:
        return "❌ Research failed. Please check your API key and try again."
    
    # Step 2: Writing
    writing_prompt = f"""
    Analyze the following research findings and create a professional summary:
    
    RESEARCH FINDINGS:
    {research_results}
    
    Create a summary that:
    - Highlights the most important information
    - Presents information in a logical flow
    - Uses clear, concise language
    - Includes key takeaways
    - Is engaging and easy to read
    
    Format: A well-structured 3-4 paragraph summary with clear headings, key insights, and actionable conclusions.
    """
    
    st.info("📝 Writing summary...")
    final_summary = writer_agent.execute_task(writing_prompt)
    
    return final_summary if final_summary else "❌ Writing failed. Please try again."

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
    
    # Get API key
    final_api_key = groq_api_key.strip() if groq_api_key and groq_api_key.strip() else os.getenv("GROQ_API_KEY")
    
    if not final_api_key:
        st.error("❌ No Groq API key provided. Please enter your API key or set GROQ_API_KEY environment variable.")
        return
    
    # Initialize Groq LLM
    groq_llm = GroqLLM(
        api_key=final_api_key,
        model=model_options[selected_model],
        temperature=temperature
    )
    
    # Create search tool instance
    search_tool = duckduckgo_search_tool
    
    # Create agents
    research_agent, writer_agent = create_agents(groq_llm, search_tool)
    
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
                    status_text.text("🔄 Initializing research...")
                    progress_bar.progress(30)
                    
                    # Execute research workflow
                    result = execute_research(query, research_agent, writer_agent, max_results)
                    
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
