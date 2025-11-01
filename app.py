# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + DUCKDUCKGO + CREWAI)
# =====================================================================

import os
import streamlit as st
from duckduckgo_search import DDGS
import requests
import json
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
# 🌐 IMPROVED DUCKDUCKGO SEARCH TOOL
# =====================================================================

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        st.write(f"🔍 Searching for: '{query}'")
        
        results = []
        with DDGS() as ddg:
            # Add timeout and better error handling
            search_results = list(ddg.text(query, max_results=max_results, region='us-en', safesearch='moderate'))
            
            if not search_results:
                st.warning("⚠️ No search results found. Trying with different parameters...")
                # Try alternative search
                search_results = list(ddg.text(query, max_results=max_results))
            
            for i, result in enumerate(search_results, 1):
                title = result.get("title", "No title")
                href = result.get("href", "No URL")
                body = result.get("body", "No description")
                
                st.write(f"📄 Result {i}: {title[:80]}...")
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {href}\n**Summary:** {body}\n")
        
        if not results:
            error_msg = f"No results found for '{query}'. The search might be rate-limited or the query might be too specific."
            st.error(f"❌ {error_msg}")
            return error_msg
            
        st.success(f"✅ Found {len(results)} search results")
        return "\n\n".join(results)
    
    except Exception as e:
        error_msg = f"Search error: {str(e)}. This might be due to rate limiting or network issues."
        st.error(f"❌ {error_msg}")
        return error_msg

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
    
    # Step 1: Perform web search with progress indication
    with st.status("🔍 Searching the web...", expanded=True) as status:
        search_results = duckduckgo_search(query, max_results)
        status.update(label="✅ Web search completed", state="complete")
    
    # Check if search actually returned useful results
    if "No results found" in search_results or "Search error" in search_results:
        st.warning("🔄 Falling back to AI knowledge base (search unavailable)...")
        
        # Use AI's internal knowledge as fallback
        fallback_prompt = f"""
        Based on your training data and knowledge, provide a comprehensive overview of: "{query}"
        
        Please include:
        - Key concepts and definitions
        - Current trends and developments
        - Important applications or use cases
        - Future outlook or predictions
        
        Format your response as a well-structured report with clear sections.
        """
        
        final_summary = groq_llm.call(
            fallback_prompt,
            system_message="You are a knowledgeable research assistant. Provide comprehensive, well-structured information based on your training data when web search is unavailable."
        )
        
        return final_summary if final_summary else "❌ Unable to generate research summary."
    
    # Step 2: Research analysis
    with st.status("📊 Analyzing search results...", expanded=True) as status:
        research_prompt = f"""
        Analyze the following web search results and extract the most important information about: "{query}"
        
        SEARCH RESULTS:
        {search_results}
        
        Please provide a comprehensive analysis that:
        1. Identifies key facts and insights from the search results
        2. Highlights the most relevant and recent information
        3. Organizes findings by importance and relevance
        4. Notes the credibility of sources where possible
        5. Extracts specific data, statistics, and trends
        
        Format your response as a detailed research report with clear sections and bullet points for key findings.
        """
        
        research_analysis = groq_llm.call(
            research_prompt,
            system_message="You are a senior research analyst. Your goal is to extract and organize the most valuable information from web search results. Be thorough, objective, and focus on factual accuracy."
        )
        status.update(label="✅ Analysis completed", state="complete")
    
    if not research_analysis:
        return "❌ Research analysis failed. Please check your API key and try again."
    
    # Step 3: Create final summary
    with st.status("✍️ Writing final summary...", expanded=True) as status:
        summary_prompt = f"""
        Based on the following research analysis, create a polished, engaging summary about: "{query}"
        
        RESEARCH ANALYSIS:
        {research_analysis}
        
        Please create a well-structured summary that:
        - Starts with an engaging introduction
        - Presents key findings in a logical flow
        - Uses clear, concise language that's easy to understand
        - Highlights the most important insights and trends
        - Includes specific examples or statistics where available
        - Ends with meaningful conclusions or takeaways
        - Is suitable for a general audience
        
        Format: 3-4 paragraphs with clear structure, engaging tone, and markdown formatting for readability.
        """
        
        final_summary = groq_llm.call(
            summary_prompt,
            system_message="You are a professional technical writer. You excel at transforming complex information into clear, engaging, and well-structured summaries that are accessible to everyone."
        )
        status.update(label="✅ Summary completed", state="complete")
    
    return final_summary if final_summary else "❌ Summary writing failed. Please try again."

# =====================================================================
# ⚙️ MAIN EXECUTION
# =====================================================================

def main():
    """Main application logic."""
    
    # User input
    query = st.text_area(
        "🔎 Enter your research topic:", 
        placeholder="e.g. Machine learning in finance 2024, Latest developments in renewable energy, Impact of AI on healthcare...",
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
                try:
                    # Execute research workflow
                    result = execute_research_workflow(query, groq_llm, max_results)
                    
                    # Display results
                    st.success("✅ Research complete!")
                    
                    # Results in expandable sections
                    with st.expander("📋 Research Report", expanded=True):
                        if result.startswith("❌") or "failed" in result.lower():
                            st.error(result)
                        else:
                            st.markdown(result)
                    
                    # Show research details
                    with st.expander("🔧 Research Details"):
                        st.write(f"**Model:** {selected_model}")
                        st.write(f"**Temperature:** {temperature}")
                        st.write(f"**Max Results:** {max_results}")
                        st.write(f"**Query:** {query}")
                        
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")

    # Display usage tips
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        - **Be specific** in your research topic for more relevant results
        - **Use current topics** for better search results
        - **Adjust temperature**: Lower (0.1-0.3) for factual topics, higher (0.7-1.0) for creative topics
        - **Use more results** for comprehensive research on complex topics
        - **Try different models** if you're not satisfied with the results
        """)

    # Troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **If search isn't working:**
        - The app will automatically fall back to AI knowledge base
        - DuckDuckGo might be rate-limited - try again in a few minutes
        - Check your internet connection
        - Try a different search query
        
        **Common issues:**
        - ❌ No search results: Usually temporary, try again later
        - 🔄 Using AI knowledge base: Search is unavailable, but AI will still help
        - ⏳ Slow responses: Groq API might be busy
        """)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + DuckDuckGo Search | " +
          "Note: Research quality depends on search results and model capabilities")

if __name__ == "__main__":
    main()
