# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + DUCKDUCKGO + CREWAI)
# =====================================================================

import os
import streamlit as st
from duckduckgo_search import DDGS
import requests
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
        results = []
        with DDGS() as ddg:
            # Try multiple search approaches if first one fails
            search_attempts = [
                query,
                f"{query} 2024",
                f"latest {query}",
                f"{query} news developments"
            ]
            
            for attempt in search_attempts:
                try:
                    search_results = list(ddg.text(attempt, max_results=max_results))
                    if search_results:
                        for result in search_results:
                            title = result.get("title", "No title")
                            href = result.get("href", "No URL")
                            body = result.get("body", "No description")
                            results.append(f"**{title}**\nURL: {href}\nDescription: {body}\n")
                        break  # Stop if we found results
                    time.sleep(1)  # Brief pause between attempts
                except Exception as e:
                    continue
            
        if not results:
            return f"No results found for '{query}'. Try using more specific search terms."
            
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        for result in results:
            # Extract URL from result string
            lines = result.split('\n')
            url_line = [line for line in lines if line.startswith('URL: ')]
            if url_line:
                url = url_line[0].replace('URL: ', '')
                if url not in seen_urls and url != "No URL":
                    seen_urls.add(url)
                    unique_results.append(result)
        
        return "\n\n".join(unique_results[:max_results])
    
    except Exception as e:
        return f"Error performing search: {str(e)}. Please try again."

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
# 🎯 IMPROVED RESEARCH WORKFLOW
# =====================================================================

def execute_research_workflow(query, groq_llm, max_results):
    """Execute the complete research workflow"""
    
    # Step 1: Perform web search with progress indicator
    search_status = st.empty()
    search_status.info("🔍 Searching the web for relevant information...")
    
    search_results = duckduckgo_search(query, max_results)
    
    # Check if search was successful
    if "No results found" in search_results or "Error performing search" in search_results:
        search_status.warning(f"⚠️ {search_results}")
        
        # If search fails, use the LLM to generate information based on its training data
        st.info("🧠 Using AI knowledge base (search unavailable)...")
        fallback_prompt = f"""
        Based on your comprehensive knowledge, provide a detailed overview of: "{query}"
        
        Please include:
        1. Key concepts and definitions
        2. Current state and recent developments
        3. Major applications and use cases
        4. Future trends and implications
        5. Important considerations or challenges
        
        Provide a well-structured, informative overview suitable for someone researching this topic.
        """
        
        fallback_result = groq_llm.call(
            fallback_prompt,
            system_message="You are an expert research assistant with comprehensive knowledge across many domains. Provide detailed, accurate, and well-structured information even without current web search results."
        )
        
        return fallback_result if fallback_result else "Unable to generate research summary. Please try a different search term."
    
    search_status.success("✅ Search completed successfully!")
    
    # Step 2: Research analysis
    analysis_status = st.empty()
    analysis_status.info("📊 Analyzing and organizing search results...")
    
    research_prompt = f"""
    Analyze the following web search results about "{query}" and create a comprehensive research summary:
    
    SEARCH RESULTS:
    {search_results}
    
    Please provide a well-organized research summary that:
    1. Starts with an executive overview
    2. Identifies and explains key findings
    3. Highlights the most important insights
    4. Organizes information logically
    5. Notes the credibility of sources where possible
    6. Ends with key takeaways
    
    Focus on creating valuable, actionable insights from the search results.
    """
    
    research_analysis = groq_llm.call(
        research_prompt,
        system_message="You are a senior research analyst. Your goal is to extract and organize the most valuable information from web search results. Be thorough, objective, and focus on factual accuracy."
    )
    
    if not research_analysis:
        analysis_status.error("❌ Research analysis failed.")
        return "Research analysis failed. Please check your API key and try again."
    
    analysis_status.success("✅ Analysis completed!")
    
    # Step 3: Create final summary
    writing_status = st.empty()
    writing_status.info("✍️ Writing final polished summary...")
    
    summary_prompt = f"""
    Transform the following research analysis into a polished, engaging final report about: "{query}"
    
    RESEARCH ANALYSIS:
    {research_analysis}
    
    Please create a professional summary that:
    - Has an engaging introduction that hooks the reader
    - Presents information in a clear, logical flow
    - Uses accessible language while maintaining accuracy
    - Highlights the most important insights prominently
    - Includes practical implications or applications
    - Ends with memorable conclusions
    
    Make it comprehensive yet concise, suitable for both experts and general audiences.
    """
    
    final_summary = groq_llm.call(
        summary_prompt,
        system_message="You are a professional technical writer and researcher. You excel at transforming complex information into clear, engaging, and well-structured reports that are accessible to everyone while maintaining accuracy and depth."
    )
    
    writing_status.success("✅ Writing completed!")
    
    return final_summary if final_summary else "Unable to generate final summary. Please try again."

# =====================================================================
# ⚙️ MAIN EXECUTION
# =====================================================================

def main():
    """Main application logic."""
    
    # User input
    query = st.text_area(
        "🔎 Enter your research topic:", 
        placeholder="e.g. Artificial Intelligence applications in healthcare, Latest developments in renewable energy, Impact of blockchain technology...",
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
                    progress_bar.progress(10)
                    
                    # Execute research workflow
                    result = execute_research_workflow(query, groq_llm, max_results)
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("✅ Research complete!")
                    
                    # Display results
                    st.success("✅ Research complete!")
                    
                    # Results in expandable sections
                    with st.expander("📋 Research Report", expanded=True):
                        if result:
                            st.markdown(result)
                        else:
                            st.error("No results generated. Please try again.")
                    
                    # Show research details
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
        - **Be specific**: Instead of "AI", try "AI applications in healthcare 2024"
        - **Use current topics**: Add "latest" or "2024" for recent information
        - **Adjust temperature**: Lower (0.1-0.3) for factual topics, higher (0.7-1.0) for creative topics
        - **Try different models**: Some models may work better for certain topics
        - **Check your query**: Make sure it's clear and well-defined
        """)

    # Search troubleshooting
    with st.expander("🔍 Search Troubleshooting"):
        st.markdown("""
        **If search fails:**
        - The app will use the AI's built-in knowledge as a fallback
        - Try more specific search terms
        - Check your internet connection
        - Try fewer max results (3-5)
        - Wait a moment and try again
        
        **Good examples:**
        - "Machine learning in finance 2024"
        - "Renewable energy advancements"
        - "Blockchain technology applications"
        """)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + DuckDuckGo Search | " +
          "Note: Research quality depends on search results and model capabilities")

if __name__ == "__main__":
    main()
