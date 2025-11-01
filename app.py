# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + DUCKDUCKGO + CREWAI)
# =====================================================================

import os
import streamlit as st
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
    🤖 **Powered by Groq + DuckDuckGo Search**  
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
        
        **Workflow:**
        - 🔍 **Research**: Searches the web for information
        - ✍️ **Writing**: Summarizes findings into a coherent report
        """)

# =====================================================================
# 🌐 DUCKDUCKGO SEARCH TOOL
# =====================================================================

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

# =====================================================================
# 🧠 GROQ LLM INTEGRATION
# =====================================================================

class GroqLLM:
    """Custom Groq LLM wrapper"""
    
    def __init__(self, api_key, model, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def call(self, prompt, system_message=None):
        """Make API call to Groq"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
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
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
        except KeyError as e:
            st.error(f"❌ Unexpected API response format: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

# =====================================================================
# 🎯 RESEARCH WORKFLOW
# =====================================================================

def execute_research_workflow(query, groq_llm, max_results):
    """Execute the complete research workflow"""
    
    # Step 1: Perform web search
    st.info("🔍 Searching the web...")
    search_results = duckduckgo_search(query, max_results)
    
    if "Error" in search_results or "No results" in search_results:
        return f"Search failed: {search_results}"
    
    # Step 2: Research analysis
    st.info("📊 Analyzing search results...")
    research_prompt = f"""
    Analyze the following web search results and extract the most important information about: "{query}"
    
    SEARCH RESULTS:
    {search_results}
    
    Please provide a comprehensive analysis that:
    1. Identifies key facts and insights
    2. Highlights the most relevant information
    3. Organizes findings by importance
    4. Notes any conflicting or complementary information
    5. Focuses on credible and authoritative sources
    
    Format your response as a detailed research report with clear sections.
    """
    
    research_analysis = groq_llm.call(
        research_prompt,
        system_message="You are a senior research analyst. Your goal is to extract and organize the most valuable information from web search results. Be thorough, objective, and focus on factual accuracy."
    )
    
    if not research_analysis:
        return "❌ Research analysis failed. Please check your API key and try again."
    
    # Step 3: Create final summary
    st.info("✍️ Writing final summary...")
    summary_prompt = f"""
    Based on the following research analysis, create a polished, engaging summary about: "{query}"
    
    RESEARCH ANALYSIS:
    {research_analysis}
    
    Please create a well-structured summary that:
    - Starts with an engaging introduction
    - Presents key findings in a logical flow
    - Uses clear, concise language that's easy to understand
    - Highlights the most important insights
    - Ends with meaningful conclusions or takeaways
    - Is suitable for a general audience
    
    Format: 3-4 paragraphs with clear structure and engaging tone.
    """
    
    final_summary = groq_llm.call(
        summary_prompt,
        system_message="You are a professional technical writer. You excel at transforming complex information into clear, engaging, and well-structured summaries that are accessible to everyone."
    )
    
    return final_summary if final_summary else "❌ Summary writing failed. Please try again."

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
        st.info("💡 Get your free API key from: https://console.groq.com")
        return
    
    # Initialize Groq LLM
    groq_llm = GroqLLM(
        api_key=final_api_key,
        model=model_options[selected_model],
        temperature=temperature
    )
    
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
                    status_text.text("🔄 Starting research workflow...")
                    progress_bar.progress(20)
                    
                    # Execute research workflow
                    result = execute_research_workflow(query, groq_llm, max_results)
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("✅ Research complete!")
                    
                    # Display results
                    st.success("✅ Research complete!")
                    
                    # Results in expandable sections
                    with st.expander("📋 Executive Summary", expanded=True):
                        if result.startswith("❌") or result.startswith("Search failed"):
                            st.error(result)
                        else:
                            st.write(result)
                    
                    # Show research details
                    with st.expander("🔧 Research Details"):
                        st.write(f"**Model:** {selected_model}")
                        st.write(f"**Temperature:** {temperature}")
                        st.write(f"**Max Results:** {max_results}")
                        st.write(f"**Query:** {query}")
                        
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    progress_bar.progress(0)
                    status_text.text("")

    # Display usage tips
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        - **Be specific** in your research topic for more relevant results
        - **Adjust temperature**: Lower (0.1-0.3) for factual topics, higher (0.7-1.0) for creative topics
        - **Use more results** for comprehensive research on complex topics
        - **Try different models** if you're not satisfied with the results
        - **Check your API key** if you get authentication errors
        """)

    # API key help
    with st.expander("🔑 API Key Help"):
        st.markdown("""
        1. **Get a free Groq API key**: Visit [https://console.groq.com](https://console.groq.com)
        2. **Create an account** and verify your email
        3. **Navigate to API Keys** in the dashboard
        4. **Create a new API key** and copy it
        5. **Paste it in the sidebar** or set as `GROQ_API_KEY` environment variable
        
        Groq offers free tier with generous limits!
        """)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + DuckDuckGo Search | " +
          "Note: Research quality depends on search results and model capabilities")

if __name__ == "__main__":
    main()
