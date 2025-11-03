# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + SERPER API)
# =====================================================================

import os
import streamlit as st
import requests
import json
from datetime import datetime

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
    🤖 **Powered by Groq + Serper API**  
    _Perform intelligent web research with reliable search results._
    """
)

# =====================================================================
# 🔑 API KEY INPUT
# =====================================================================

with st.sidebar:
    st.header("🔐 Configuration")
    
    # API Key inputs
    groq_api_key = st.text_input(
        "Enter your Groq API Key", 
        type="password",
        placeholder="gsk_... or leave blank to use GROQ_API_KEY env var"
    )
    
    serper_api_key = st.text_input(
        "Serper API Key (Required)",
        type="password",
        placeholder="Enter your Serper API key",
        help="Get free key from https://serper.dev"
    )
    
    # Model selection
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
    st.info("💡 **Using Serper API for reliable web search**")

# =====================================================================
# 🔍 SERPER API SEARCH IMPLEMENTATION
# =====================================================================

def serper_search(query: str, max_results: int = 5, api_key: str = None) -> str:
    """Search using Serper API - our primary search method"""
    if not api_key:
        return "❌ Serper API key not provided"
    
    try:
        st.write(f"🔍 Searching Serper API for: '{query}'")
        
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query, 
            "num": max_results,
            "gl": "us",  # Country: United States
            "hl": "en"   # Language: English
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        with st.spinner("Searching the web..."):
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Process organic results
        if 'organic' in data and data['organic']:
            for i, result in enumerate(data['organic'][:max_results], 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                
                # Show real-time progress
                st.write(f"✅ Found: {title[:80]}...")
        
        # Also check for news results if organic results are limited
        if len(results) < max_results and 'news' in data and data['news']:
            news_to_add = max_results - len(results)
            for i, result in enumerate(data['news'][:news_to_add], len(results) + 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📰 News {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                st.write(f"✅ Found news: {title[:80]}...")
        
        if results:
            st.success(f"✅ Serper API found {len(results)} high-quality results")
            return "\n\n".join(results)
        else:
            st.warning("⚠️ No search results found via Serper API")
            return "❌ No search results found"
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Serper API request failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error with Serper API: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"❌ {error_msg}"

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
            st.error(f"❌ Groq API Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

# =====================================================================
# 🎯 RESEARCH WORKFLOW
# =====================================================================

def execute_research_workflow(query, groq_llm, max_results, serper_key):
    """Execute the complete research workflow using only Serper API"""
    
    # Step 1: Perform web search with Serper API
    with st.status("🔍 Searching the web with Serper API...", expanded=True) as status:
        search_results = serper_search(query, max_results, serper_key)
        
        # Check if search was successful
        if search_results.startswith("❌"):
            st.error("Search failed. Please check your Serper API key and try again.")
            return None
            
        status.update(label="✅ Web search completed", state="complete")
    
    # Step 2: Analyze search results and generate research report
    with st.status("📊 Analyzing search results and generating report...", expanded=True) as status:
        research_prompt = f"""
        Analyze the following web search results and create a comprehensive research report about: "{query}"
        
        SEARCH RESULTS:
        {search_results}
        
        Current Date: {datetime.now().strftime('%Y-%m-%d')}
        
        Please create a well-structured research report that includes:
        
        ## Executive Summary
        - Brief overview of key findings
        - Main conclusions and insights
        
        ## Key Findings
        - Most important information from the search results
        - Relevant data, statistics, and facts
        - Current trends and developments
        
        ## Detailed Analysis
        - In-depth analysis of the topic based on the search results
        - Connections between different pieces of information
        - Credibility assessment of sources where possible
        
        ## Applications and Implications
        - Practical applications of the information
        - Potential impact and significance
        - Future implications
        
        ## Conclusion
        - Summary of main points
        - Final recommendations or insights
        
        Base your report strictly on the provided search results. Be comprehensive, objective, and well-organized.
        """
        
        system_msg = """You are a professional research analyst. Create comprehensive, 
        well-structured research reports based on web search results. Be factual, objective, 
        and focus on synthesizing information from the provided sources."""
        
        research_report = groq_llm.call(research_prompt, system_msg)
        status.update(label="✅ Research report completed", state="complete")
    
    if not research_report:
        return None
    
    # Step 3: Linkedin Post
    with st.status("💬 Creating LinkedIn post...", expanded=True) as status:
        linkedin_post_prompt = f"""
        Based on the following research report, create a high-quality LinkedIn post that summarizes and humanizes the key findings.

        RESEARCH REPORT:
        {research_report}

        Write a professional, engaging LinkedIn post that:
        - Has a catchy headline with a relevant emoji (e.g., 🚀, 🌾, 💡, 🔍).
        - Opens with a strong hook that sparks curiosity or emotion.
        - Organizes content into 3–6 short sections, each with a short bold title or emoji header (e.g., 🌾 The New Farming Frontier).
        - Explains key insights, data, and implications clearly, in natural, conversational English.
        - Feels informative but not academic — it should sound like something a human professional would share to educate or inspire their audience.
        - Closes with a reflective or thought-provoking question that invites engagement (e.g., “What are your thoughts?” or “Have you seen this trend in your industry?”).
        - Includes 5–8 relevant hashtags at the end.
        - Optionally includes emojis throughout the text for readability and visual appeal.
        - Length: 1,000–1,500 words maximum. Rich, insightful, but easy to read.

        Format it naturally for LinkedIn (line breaks between sections, no Markdown or bullet points).
        """

        linkedin_post = groq_llm.call(
            linkedin_post_prompt,
            "You are a professional LinkedIn storyteller. Write educational, engaging, and insightful posts based on research findings. Your tone should be human, authentic, and thought-provoking — designed to inform and connect with professionals."
        )

        status.update(label="✅ LinkedIn post created successfully", state="complete")

# =====================================================================
# ⚙️ MAIN EXECUTION
# =====================================================================

def main():
    """Main application logic."""
    
    # User input
    query = st.text_area(
        "🔎 Enter your research topic:", 
        placeholder="e.g. Machine learning in Agriculture 2024, Renewable energy trends, AI in healthcare applications...",
        height=100
    )
    
    # Get API keys
    final_groq_key = groq_api_key.strip() if groq_api_key and groq_api_key.strip() else os.getenv("GROQ_API_KEY")
    final_serper_key = serper_api_key.strip() if serper_api_key and serper_api_key.strip() else os.getenv("SERPER_API_KEY")
    
    # Validate API keys
    if not final_groq_key:
        st.error("❌ No Groq API key provided. Please enter your API key or set GROQ_API_KEY environment variable.")
        st.info("💡 Get your free API key from: https://console.groq.com")
        return
        
    if not final_serper_key:
        st.error("❌ No Serper API key provided. Please enter your Serper API key.")
        st.info("💡 Get your free API key from: https://serper.dev")
        return
    
    # Initialize Groq LLM
    groq_llm = GroqLLM(
        api_key=final_groq_key,
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
                    result = execute_research_workflow(
                        query, groq_llm, max_results, final_serper_key
                    )
                    
                    if result is None:
                        st.error("❌ Research failed. Please check your API keys and try again.")
                    else:
                        st.success("✅ Research complete!")
                        
                        # Display results
                        with st.expander("📋 LinkedIn Post", expanded=True):
                            st.markdown(result["linkedin_post"])
                        
                        with st.expander("📊 Full Research Report", expanded=False):
                            st.markdown(result["research_report"])
                        
                        with st.expander("🔍 Raw Search Results", expanded=False):
                            st.markdown(result["search_results"])
                        
                        with st.expander("🔧 Research Details", expanded=False):
                            st.write(f"**Model:** {selected_model}")
                            st.write(f"**Temperature:** {temperature}")
                            st.write(f"**Max Results:** {max_results}")
                            st.write(f"**Search Engine:** Serper API")
                            st.write(f"**Query:** {query}")
                            st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")

    # API Key Help Section
    with st.expander("🔑 API Key Information"):
        st.markdown("""
        **Required API Keys:**
        
        **1. Serper API** (Web Search)
        - Visit: https://serper.dev/
        - Free tier: 2,500 searches/month
        - Fast and reliable Google search results
        - No rate limiting issues
        
        **2. Groq API** (AI Processing)
        - Visit: https://console.groq.com/
        - Free tier: Generous limits
        - Fast inference speeds
        - Multiple model options
        
        **How to get started:**
        1. Sign up for free accounts at both services
        2. Copy your API keys from the dashboards
        3. Enter both keys in the sidebar
        4. Start researching!
        """)

    # Tips Section
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        **For optimal research results:**
        
        - **Be specific** with your search queries
        - **Use current year** for time-sensitive topics
        - **Include industry/domain** for focused results
        - **Adjust temperature**: Lower for facts, higher for creativity
        - **Use more results** for comprehensive research
        
        **Example queries:**
        - "Machine learning applications in agriculture 2024"
        - "Renewable energy adoption trends in Europe"
        - "AI in healthcare diagnostics latest developments"
        """)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + Serper API | " +
          "Reliable web search with AI-powered analysis")

if __name__ == "__main__":
    main()
