# =====================================================================
# 🚀 STREAMLIT AGENTIC RESEARCH APP (GROQ + SERPER API)
# =====================================================================

import os
import streamlit as st
import requests
import json
from datetime import datetime
from io import BytesIO
from docx import Document

# =====================================================================
# ⚙️ APP CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Groq Agentic Researcher",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Responsive UI Styling ---
st.markdown("""
    <style>
        .main {
            padding: 1.2rem;
        }
        @media (min-width: 1024px) {
            .main > div {
                max-width: 1000px;
                margin: auto;
            }
        }
        @media (max-width: 768px) {
            .main {
                padding: 0.8rem;
            }
        }
        h1, h2, h3 {
            color: #222;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Groq-Powered Agentic Researcher")
st.markdown(
    """
    🤖 Powered by Groq + Serper API  
    Perform intelligent web research with reliable search results.
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
            "gl": "us",
            "hl": "en"
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

    # Step 1: Web search
    with st.status("🔍 Searching the web with Serper API...", expanded=True) as status:
        search_results = serper_search(query, max_results, serper_key)
        if search_results.startswith("❌"):
            st.error("Search failed. Please check your Serper API key and try again.")
            return None
        status.update(label="✅ Web search completed", state="complete")

    # Step 2: Generate research report
    with st.status("📊 Analyzing search results and generating report...", expanded=True) as status:
        research_prompt = f"""
        Analyze the following web search results and create a comprehensive research report about: "{query}"
        
        SEARCH RESULTS:
        {search_results}
        Current Date: {datetime.now().strftime('%Y-%m-%d')}
        Please create a well-structured research report...
        """
        system_msg = "You are a professional research analyst..."
        research_report = groq_llm.call(research_prompt, system_msg)
        status.update(label="✅ Research report completed", state="complete")

    if not research_report:
        return None

    # Step 3: LinkedIn Post
    with st.status("💬 Creating LinkedIn post...", expanded=True) as status:
        linkedin_post_prompt = f"""
        Based on the following research report, create a professional LinkedIn post...
        RESEARCH REPORT:
        {research_report}
        """
        linkedin_post = groq_llm.call(linkedin_post_prompt, "You are a professional LinkedIn storyteller.")
        status.update(label="✅ LinkedIn post created successfully", state="complete")

    return {
        "search_results": search_results,
        "research_report": research_report,
        "linkedin_post": linkedin_post
    }

# =====================================================================
# ⚙️ MAIN EXECUTION
# =====================================================================

def main():
    """Main application logic."""

    query = st.text_area("🔎 Enter your research topic:", height=100)

    final_groq_key = groq_api_key.strip() if groq_api_key else os.getenv("GROQ_API_KEY")
    final_serper_key = serper_api_key.strip() if serper_api_key else os.getenv("SERPER_API_KEY")

    if not final_groq_key:
        st.error("❌ No Groq API key provided.")
        return
    if not final_serper_key:
        st.error("❌ No Serper API key provided.")
        return

    groq_llm = GroqLLM(final_groq_key, model_options[selected_model], temperature)

    if st.button("🚀 Run Research", use_container_width=True, type="primary"):
        if not query.strip():
            st.warning("⚠️ Please enter a research topic.")
        else:
            result = execute_research_workflow(query, groq_llm, max_results, final_serper_key)
            if result:
                st.success("✅ Research complete!")

                with st.expander("📋 LinkedIn Post", expanded=True):
                    st.markdown(result["linkedin_post"])

                    # --- Download Buttons (NEW) ---
                    txt_data = BytesIO(result["linkedin_post"].encode('utf-8'))
                    st.download_button("⬇️ Download as .TXT", txt_data,
                                       file_name=f"{query.replace(' ', '_')}_LinkedIn_Post.txt",
                                       mime="text/plain")

                    doc = Document()
                    doc.add_paragraph(result["linkedin_post"])
                    docx_stream = BytesIO()
                    doc.save(docx_stream)
                    docx_stream.seek(0)
                    st.download_button("⬇️ Download as .DOCX", docx_stream,
                                       file_name=f"{query.replace(' ', '_')}_LinkedIn_Post.docx",
                                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

                with st.expander("📊 Full Research Report", expanded=False):
                    st.markdown(result["research_report"])

                with st.expander("🔍 Raw Search Results", expanded=False):
                    st.markdown(result["search_results"])

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + Serper API | Reliable web search with AI-powered analysis")

if __name__ == "__main__":
    main()
