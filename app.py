# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + MULTI-SEARCH + CREWAI)
# =====================================================================

import os
import streamlit as st
import requests
import json
import time
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
    🤖 **Powered by Groq + Multiple Search Methods**  
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
    
    # Optional: Serper API key for enhanced search
    serper_api_key = st.text_input(
        "Optional: Serper API Key (for better search)",
        type="password",
        placeholder="Leave blank for free search methods"
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
    
    # Search method selection
    search_method = st.radio(
        "Search Method:",
        ["Auto (Try Multiple)", "DuckDuckGo", "Serper API", "AI Knowledge Only"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Free tier available for all methods!**")

# =====================================================================
# 🔍 MULTI-SEARCH ENGINE IMPLEMENTATION
# =====================================================================

def serper_search(query: str, max_results: int = 5, api_key: str = None) -> str:
    """Search using Serper API (more reliable)"""
    if not api_key:
        return "Serper API key not provided"
    
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": max_results})
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Process organic results
        if 'organic' in data:
            for i, result in enumerate(data['organic'][:max_results], 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
        
        if results:
            st.success(f"✅ Serper found {len(results)} results")
            return "\n\n".join(results)
        else:
            return "No results found via Serper"
            
    except Exception as e:
        return f"Serper search error: {str(e)}"

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search using DuckDuckGo with enhanced error handling"""
    try:
        # Try to import and use DuckDuckGo
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddg:
            search_results = list(ddg.text(
                query, 
                max_results=max_results,
                region='wt-wt',
                safesearch='moderate',
                timelimit='y'
            ))
            
            for i, result in enumerate(search_results, 1):
                title = result.get("title", "No title")
                href = result.get("href", "No URL")
                body = result.get("body", "No description")
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {href}\n**Summary:** {body}\n")
        
        if results:
            st.success(f"✅ DuckDuckGo found {len(results)} results")
            return "\n\n".join(results)
        else:
            return "No results found via DuckDuckGo"
    
    except Exception as e:
        return f"DuckDuckGo search error: {str(e)}"

def brave_search(query: str, max_results: int = 5) -> str:
    """Alternative search method using Brave Search (free tier available)"""
    try:
        # This is a placeholder - you would need to sign up for Brave Search API
        # For now, we'll simulate this or use another method
        return "Brave Search requires API key setup"
    except Exception as e:
        return f"Brave search error: {str(e)}"

def perform_web_search(query: str, max_results: int = 5, method: str = "auto", serper_key: str = None) -> str:
    """Main search function that tries multiple methods"""
    
    st.write(f"🔍 Searching for: '{query}'")
    
    if method == "ai knowledge only":
        return "search_unavailable"
    
    search_attempts = []
    
    # Try Serper first if API key is available
    if serper_key and method in ["auto", "Serper API"]:
        st.write("🔄 Trying Serper API...")
        result = serper_search(query, max_results, serper_key)
        if "found" in result.lower() and "no results" not in result.lower():
            return result
        search_attempts.append(f"Serper: {result}")
    
    # Try DuckDuckGo
    if method in ["auto", "DuckDuckGo"]:
        st.write("🔄 Trying DuckDuckGo...")
        result = duckduckgo_search(query, max_results)
        if "found" in result.lower() and "no results" not in result.lower():
            return result
        search_attempts.append(f"DuckDuckGo: {result}")
    
    # If all methods fail
    error_msg = f"All search methods failed. Attempts:\n" + "\n".join(search_attempts)
    st.error("❌ " + error_msg)
    return "search_unavailable"

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
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

# =====================================================================
# 🎯 RESEARCH WORKFLOW
# =====================================================================

def execute_research_workflow(query, groq_llm, max_results, search_method, serper_key):
    """Execute the complete research workflow"""
    
    # Step 1: Perform web search
    with st.status("🔍 Searching the web...", expanded=True) as status:
        search_results = perform_web_search(query, max_results, search_method, serper_key)
        status.update(label="✅ Search completed", state="complete")
    
    # Step 2: Generate research report
    with st.status("📊 Generating research report...", expanded=True) as status:
        if search_results == "search_unavailable":
            # Use AI's internal knowledge with enhanced prompt
            st.info("🧠 Using AI knowledge base with enhanced research...")
            research_prompt = f"""
            Create a comprehensive research report about: "{query}"
            
            Current Date: {datetime.now().strftime('%Y-%m-%d')}
            
            Please provide a detailed analysis including:
            
            ## Executive Summary
            - Key overview and importance
            
            ## Current Trends and Developments (2023-2024)
            - Latest advancements and innovations
            - Market trends and adoption rates
            - Key players and organizations
            
            ## Applications and Use Cases
            - Practical implementations
            - Industry-specific applications
            - Real-world examples
            
            ## Challenges and Opportunities
            - Current limitations and barriers
            - Future growth potential
            - Emerging opportunities
            
            ## Future Outlook
            - Predictions and forecasts
            - Potential impact
            - Recommended areas for further research
            
            Format this as a professional research report with clear sections and bullet points.
            Focus on providing actionable insights and comprehensive coverage.
            """
            
            system_msg = """You are a senior research analyst at a top consulting firm. 
            Create comprehensive, well-structured research reports that are data-driven and insightful. 
            Even without web access, draw upon your extensive training data to provide valuable analysis."""
            
        else:
            # Use actual search results
            research_prompt = f"""
            Analyze the following web search results and create a comprehensive research report about: "{query}"
            
            SEARCH RESULTS:
            {search_results}
            
            Current Date: {datetime.now().strftime('%Y-%m-%d')}
            
            Please create a structured research report that:
            1. Synthesizes information from the search results
            2. Identifies key trends and patterns
            3. Highlights credible sources and data points
            4. Provides balanced analysis of different perspectives
            5. Includes specific examples and statistics where available
            
            Format as a professional report with clear sections and evidence-based insights.
            """
            
            system_msg = "You are a senior research analyst. Synthesize search results into comprehensive, well-structured reports with credible insights."
        
        research_report = groq_llm.call(research_prompt, system_msg)
        status.update(label="✅ Research report completed", state="complete")
    
    if not research_report:
        return "❌ Research failed. Please check your API key and try again."
    
    # Step 3: Create executive summary
    with st.status("✍️ Creating executive summary...", expanded=True) as status:
        summary_prompt = f"""
        Based on the following research report, create a concise executive summary:
        
        RESEARCH REPORT:
        {research_report}
        
        Create a 2-3 paragraph executive summary that:
        - Highlights the most important findings
        - Presents key insights in a business-friendly format
        - Includes actionable recommendations
        - Is suitable for busy executives
        
        Keep it concise, impactful, and easy to understand.
        """
        
        executive_summary = groq_llm.call(
            summary_prompt,
            "You are an expert business communicator. Create clear, concise executive summaries that highlight key insights and recommendations."
        )
        status.update(label="✅ Executive summary completed", state="complete")
    
    return {
        "research_report": research_report,
        "executive_summary": executive_summary,
        "search_used": search_results != "search_unavailable"
    }

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
    
    if not final_groq_key:
        st.error("❌ No Groq API key provided. Please enter your API key or set GROQ_API_KEY environment variable.")
        st.info("💡 Get your free API key from: https://console.groq.com")
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
                        query, groq_llm, max_results, search_method.lower(), final_serper_key
                    )
                    
                    if isinstance(result, str) and result.startswith("❌"):
                        st.error(result)
                    else:
                        st.success("✅ Research complete!")
                        
                        # Display results
                        with st.expander("📋 Executive Summary", expanded=True):
                            st.markdown(result["executive_summary"])
                        
                        with st.expander("📊 Full Research Report", expanded=False):
                            st.markdown(result["research_report"])
                        
                        with st.expander("🔧 Research Details", expanded=False):
                            st.write(f"**Model:** {selected_model}")
                            st.write(f"**Temperature:** {temperature}")
                            st.write(f"**Max Results:** {max_results}")
                            st.write(f"**Search Method:** {search_method}")
                            st.write(f"**Web Search Used:** {result['search_used']}")
                            st.write(f"**Query:** {query}")
                        
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")

    # API Key Help Section
    with st.expander("🔑 Get API Keys (Free Tiers Available)"):
        st.markdown("""
        **🌐 For Better Search Results:**
        
        **1. Serper API (Recommended)**
        - Visit: https://serper.dev/
        - Free tier: 2,500 searches/month
        - More reliable than DuckDuckGo
        - Easy setup
        
        **2. Groq API (Required)**
        - Visit: https://console.groq.com/
        - Free tier: Generous limits
        - Fast inference speeds
        
        **Setup:**
        - Get your free API keys
        - Enter Groq API key in sidebar
        - Optional: Enter Serper API key for better search
        """)

    # Troubleshooting Section
    with st.expander("🔧 Search Troubleshooting"):
        st.markdown("""
        **If search isn't working:**
        
        ✅ **Quick Fixes:**
        - Try **"AI Knowledge Only"** mode in sidebar
        - Get a **free Serper API key** (more reliable)
        - Use more **specific search terms**
        - Try again in **5-10 minutes** (rate limits reset)
        
        🔄 **Search Methods Available:**
        1. **Serper API** (Most reliable - needs free API key)
        2. **DuckDuckGo** (Free but often rate-limited)
        3. **AI Knowledge** (Always works - uses model's training data)
        
        💡 **Pro Tip:** Get a free Serper API key for the best experience!
        """)

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + Multiple Search Methods | " +
          "Free tiers available for all services")

if __name__ == "__main__":
    main()
