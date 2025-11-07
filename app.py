# =====================================================================
# 🚀 ENHANCED STREAMLIT AGENTIC RESEARCH APP (GROQ + SERPER API)
# =====================================================================

import os
import streamlit as st
import requests
import json
from datetime import datetime
from io import BytesIO
from docx import Document
from functools import lru_cache
import time
from streamlit_copy import st_copy_button

# =====================================================================
# ⚙️ APP CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="LinkedIn Post Generator",
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
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 LinkedIn Post Generator")
st.markdown(
    """
    🤖 Powered by Groq + Serper API  
    Perform intelligent web research with reliable search results and Generate a LinkedIn Post.
    """
)

# =====================================================================
# 🎯 CONSTANTS & PROMPT TEMPLATES
# =====================================================================

LINKEDIN_POST_PROMPT = """
Create a professional LinkedIn post based on this research. Follow these guidelines:

**STRUCTURE:**
🎯 Catchy Headline with Emoji
💡 Strong Opening Hook
📊 Key Insights (3-5 bullet points)
🚀 Practical Implications
🤔 Thought-provoking Question
🏷️ Relevant Hashtags

**TONE:**
- Professional yet conversational
- Data-driven but accessible
- Engaging and shareable
- Authentic voice

**FORMATTING:**
- Use emojis sparingly for visual breaks
- Short paragraphs (2-3 lines max)
- Clear section separation
- Mobile-friendly formatting

Research Content: {research_content}

Generate the LinkedIn post accordingly:
"""

# =====================================================================
# 🔑 API KEY INPUT & VALIDATION
# =====================================================================

def validate_api_keys(groq_key, serper_key):
    """Validate API keys before proceeding"""
    errors = []
    if not groq_key:
        errors.append("❌ Groq API key is required")
    elif not groq_key.startswith(('gsk_', 'gpk_')):
        errors.append("❌ Invalid Groq API key format (should start with gsk_ or gpk_)")
    
    if not serper_key:
        errors.append("❌ Serper API key is required")
    elif len(serper_key) < 10:
        errors.append("❌ Invalid Serper API key")
    
    return errors

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
    st.header("⚙️ Advanced Settings")
    
    # Post customization
    post_tone = st.selectbox(
        "Post Tone",
        ["Professional", "Conversational", "Inspirational", "Technical", "Casual"]
    )
    
    include_statistics = st.checkbox("Include Statistics", value=True)
    add_call_to_action = st.checkbox("Add Call-to-Action", value=True)
    
    # Target audience
    audience_options = ["Executives", "Managers", "Technical", "General", "Students", "Entrepreneurs"]
    audience = st.multiselect(
        "Target Audience",
        audience_options,
        default=["General"]
    )

    st.markdown("---")
    st.info("💡 **Using Serper API for reliable web search**")

# =====================================================================
# 🔍 ENHANCED SERPER API SEARCH IMPLEMENTATION
# =====================================================================

def safe_serper_search(query: str, max_results: int = 5, api_key: str = None):
    """Enhanced search with better error handling and validation"""
    if not api_key:
        return {"formatted_results": "❌ Serper API key not provided", "urls": []}
    
    # Query validation
    if not query or len(query.strip()) < 2:
        return {"formatted_results": "❌ Search query too short", "urls": []}
    
    if len(query) > 300:
        return {"formatted_results": "❌ Search query too long (max 300 characters)", "urls": []}
    
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
        url_list = []

        # Process organic results
        if 'organic' in data and data['organic']:
            for i, result in enumerate(data['organic'][:max_results], 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📄 Result {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                url_list.append(link)
                st.write(f"✅ Found: {title[:80]}...")

        # Also check for news results if organic results are limited
        if len(results) < max_results and 'news' in data and data['news']:
            news_to_add = max_results - len(results)
            for i, result in enumerate(data['news'][:news_to_add], len(results) + 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No URL')
                snippet = result.get('snippet', 'No description')
                results.append(f"### 📰 News {i}: {title}\n**URL:** {link}\n**Summary:** {snippet}\n")
                url_list.append(link)
                st.write(f"✅ Found news: {title[:80]}...")

        if results:
            st.success(f"✅ Serper API found {len(results)} high-quality results")
            return {
                "formatted_results": "\n\n".join(results),
                "urls": url_list
            }
        else:
            st.warning("⚠️ No search results found via Serper API")
            return {
                "formatted_results": "❌ No search results found",
                "urls": []
            }

    except requests.exceptions.RequestException as e:
        error_msg = f"Serper API request failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        return {
            "formatted_results": f"❌ {error_msg}",
            "urls": []
        }
    except Exception as e:
        error_msg = f"Unexpected error with Serper API: {str(e)}"
        st.error(f"❌ {error_msg}")
        return {
            "formatted_results": f"❌ {error_msg}",
            "urls": []
        }

@lru_cache(maxsize=100)
def cached_search(query: str, max_results: int, api_key: str):
    """Cache search results to avoid duplicate API calls"""
    return safe_serper_search(query, max_results, api_key)

# =====================================================================
# 🧠 ENHANCED GROQ LLM INTEGRATION WITH RATE LIMITING
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

class RateLimitedGroqLLM:
    """Enhanced Groq LLM with rate limiting"""
    
    def __init__(self, api_key, model, temperature=0.7, requests_per_minute=30):
        self.llm = GroqLLM(api_key, model, temperature)
        self.requests_per_minute = requests_per_minute
        self.last_call_time = 0
        self.min_interval = 60.0 / requests_per_minute

    def call(self, prompt, system_message=None):
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            with st.spinner(f"Rate limiting... waiting {sleep_time:.1f}s"):
                time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        return self.llm.call(prompt, system_message)

# =====================================================================
# 📱 FACEBOOK & WHATSAPP POST GENERATION
# =====================================================================

def generate_facebook_post(linkedin_post, groq_llm):
    """Generate Facebook version from LinkedIn post"""
    facebook_prompt = f"""
    Transform this LinkedIn post into an engaging Facebook post:
    
    KEY REQUIREMENTS:
    - SHORTER: 200-400 characters max
    - CONVERSATIONAL & FRIENDLY tone
    - Use emojis naturally
    - Keep the core message but make it more personal
    - End with a question to encourage engagement
    - Include 3-5 relevant hashtags
    
    LINKEDIN POST:
    {linkedin_post}
    
    Return ONLY the Facebook post text, nothing else.
    """
    
    facebook_post = groq_llm.call(
        facebook_prompt,
        "You are a social media expert who specializes in adapting professional content for Facebook's friendly, conversational audience."
    )
    
    return facebook_post if facebook_post else "Facebook post generation failed"

def generate_whatsapp_hook(linkedin_post, groq_llm):
    """Generate ultra-short WhatsApp teaser"""
    whatsapp_prompt = f"""
    Create a SUPER SHORT WhatsApp teaser from this LinkedIn post:
    
    KEY REQUIREMENTS:
    - 1-3 lines MAX (very concise)
    - Intriguing hook that makes people want to read more
    - Casual, conversational tone
    - End with: 🔗 Read full post: [LinkedIn URL]
    
    LINKEDIN POST:
    {linkedin_post}
    
    Return ONLY the WhatsApp message text, nothing else.
    """
    
    whatsapp_hook = groq_llm.call(
        whatsapp_prompt,
        "You are a messaging expert who creates compelling, ultra-short teasers that drive clicks."
    )
    
    return whatsapp_hook if whatsapp_hook else "WhatsApp hook generation failed"

# =====================================================================
# 🎯 ENHANCED RESEARCH WORKFLOW WITH PROGRESS TRACKING
# =====================================================================

def execute_research_workflow_with_progress(query, groq_llm, max_results, serper_key, post_tone, audience, include_stats, add_cta):
    """Enhanced workflow with progress tracking and customization"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Web search
    status_text.text("🔍 Searching the web...")
    search_data = cached_search(query, max_results, serper_key)
    progress_bar.progress(25)
    
    if search_data["formatted_results"].startswith("❌"):
        st.error("Search failed. Please check your query and API key.")
        return None

    # Step 2: Research report
    status_text.text("📊 Analyzing results and generating research report...")
    
    research_prompt = f"""
    Analyze these search results and create a comprehensive research report about: "{query}"
    
    SEARCH RESULTS:
    {search_data["formatted_results"]}
    
    Current Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Create a well-structured research report with:
    1. EXECUTIVE SUMMARY: Brief overview of key findings
    2. KEY FINDINGS: Main insights and discoveries
    3. DATA & STATISTICS: {"Include relevant data points and statistics" if include_stats else "Focus on qualitative insights"}
    4. IMPLICATIONS: What these findings mean in practice
    5. RECOMMENDATIONS: Actionable suggestions
    
    Target Audience: {', '.join(audience)}
    Tone: {post_tone.lower()}
    """
    
    research_report = groq_llm.call(research_prompt, "You are a professional research analyst.")
    progress_bar.progress(60)

    if not research_report:
        st.error("❌ Failed to generate research report")
        return None

    # Step 3: LinkedIn post
    status_text.text("💬 Creating LinkedIn post...")
    
    enhanced_linkedin_prompt = f"""
    {LINKEDIN_POST_PROMPT.format(research_content=research_report)}
    
    ADDITIONAL REQUIREMENTS:
    - Tone: {post_tone}
    - Target Audience: {', '.join(audience)}
    - {"Include relevant statistics and data" if include_stats else "Focus on conceptual insights"}
    - {"Add a clear call-to-action" if add_cta else "End with a thought-provoking question"}
    - Make it engaging for {post_tone.lower()} professional audience
    """
    
    linkedin_post = groq_llm.call(
        enhanced_linkedin_prompt,
        "You are a professional LinkedIn content creator with expertise in creating viral professional content. Your post must be of length 1,200 to 1,500 words"
    )
    
    # Generate Facebook and WhatsApp versions
    facebook_post = generate_facebook_post(linkedin_post, groq_llm)
    whatsapp_hook = generate_whatsapp_hook(linkedin_post, groq_llm)
    
    progress_bar.progress(100)
    status_text.text("✅ Research complete!")
    
    return {
        "search_results": search_data["formatted_results"],
        "search_urls": search_data["urls"],
        "research_report": research_report,
        "linkedin_post": linkedin_post,
        "facebook_post": facebook_post,
        "whatsapp_hook": whatsapp_hook
    }

# =====================================================================
# ⚙️ MAIN EXECUTION WITH SESSION STATE
# =====================================================================

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

if 'last_results' not in st.session_state:
    st.session_state.last_results = None

def save_to_history(query, results):
    """Save research results to session history"""
    history_item = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'linkedin_post': results['linkedin_post'],
        'research_report': results['research_report']
    }
    st.session_state.research_history.append(history_item)
    st.session_state.last_results = results

def main():
    """Main application logic."""
    
    query = st.text_area("🔎 Enter your research topic:", height=100, placeholder="e.g., The impact of AI on digital marketing in 2024...")

    final_groq_key = groq_api_key.strip() if groq_api_key else os.getenv("GROQ_API_KEY")
    final_serper_key = serper_api_key.strip() if serper_api_key else os.getenv("SERPER_API_KEY")

    # Validate API keys
    validation_errors = validate_api_keys(final_groq_key, final_serper_key)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        return

    # Initialize enhanced LLM with rate limiting
    groq_llm = RateLimitedGroqLLM(final_groq_key, model_options[selected_model], temperature)

    # Research execution
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Run Research", use_container_width=True, type="primary"):
            if not query.strip():
                st.warning("⚠️ Please enter a research topic.")
            elif len(query.strip()) < 5:
                st.warning("⚠️ Please enter a more detailed research topic (at least 5 characters).")
            else:
                result = execute_research_workflow_with_progress(
                    query, 
                    groq_llm, 
                    max_results, 
                    final_serper_key,
                    post_tone,
                    audience,
                    include_statistics,
                    add_call_to_action
                )
                if result:
                    save_to_history(query, result)
                    st.success("✅ Research complete!")
                    st.session_state.last_results = result

    with col2:
        if st.session_state.research_history:
            if st.button("📚 View Research History", use_container_width=True):
                st.session_state.show_history = True

    # Display results if available
    if st.session_state.last_results:
        result = st.session_state.last_results
        
        # Create tabs for different content types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💼 LinkedIn", "📱 Facebook", "💬 WhatsApp", "🔍 URLs", "📊 Research"])
        
        with tab1:
            st.subheader("💼 LinkedIn Post")
            st.markdown(result["linkedin_post"])
            
            # Copy button for LinkedIn Post
            st_copy_button(result["linkedin_post"], "📋 Copy LinkedIn Post")
            
            # Text area for easy selection
            st.text_area(
                "LinkedIn Post Content", 
                value=result["linkedin_post"],
                height=300,
                key="linkedin_post_area",
                label_visibility="collapsed"
            )

        with tab2:
            st.subheader("📱 Facebook Post")
            st.markdown(result["facebook_post"])
            
            # Copy button for Facebook Post
            st_copy_button(result["facebook_post"], "📋 Copy Facebook Post")
            
            # Text area for easy selection
            st.text_area(
                "Facebook Post Content", 
                value=result["facebook_post"],
                height=200,
                key="facebook_post_area",
                label_visibility="collapsed"
            )

        with tab3:
            st.subheader("💬 WhatsApp Hook")
            st.markdown(result["whatsapp_hook"])
            
            # Copy button for WhatsApp Hook
            st_copy_button(result["whatsapp_hook"], "📋 Copy WhatsApp Hook")
            
            # Text area for easy selection
            st.text_area(
                "WhatsApp Hook Content", 
                value=result["whatsapp_hook"],
                height=150,
                key="whatsapp_hook_area",
                label_visibility="collapsed"
            )

        with tab4:
            st.subheader("🔍 Research URLs")
            if result["search_urls"]:
                urls_text = "\n".join(result["search_urls"])
                st.markdown(f"**Found {len(result['search_urls'])} URLs:**")
                
                # Copy button for all URLs
                st_copy_button(urls_text, "📋 Copy All URLs")
                
                # Text area for URLs
                st.text_area(
                    "Research URLs", 
                    value=urls_text,
                    height=200,
                    key="urls_area",
                    label_visibility="collapsed"
                )
                
                # Individual URL copy buttons
                st.markdown("**Individual URLs:**")
                for i, url in enumerate(result["search_urls"], 1):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"`{i}. {url}`")
                    with col2:
                        st_copy_button(url, "📋")

        with tab5:
            st.subheader("📊 Research Report")
            st.markdown(result["research_report"])
            
            # Copy button for Research Report
            st_copy_button(result["research_report"], "📋 Copy Research Report")
            
            # Text area for easy selection
            st.text_area(
                "Research Report Content", 
                value=result["research_report"],
                height=400,
                key="research_report_area", 
                label_visibility="collapsed"
            )

    # Research history view
    if st.session_state.get('show_history', False) and st.session_state.research_history:
        st.markdown("---")
        st.subheader("📚 Research History")
        
        for i, item in enumerate(reversed(st.session_state.research_history[-5:]), 1):
            with st.expander(f"#{i}: {item['query']} - {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')}"):
                st.markdown("**LinkedIn Post:**")
                st.markdown(item['linkedin_post'][:500] + "..." if len(item['linkedin_post']) > 500 else item['linkedin_post'])
                
                if st.button(f"Load this result", key=f"load_{i}"):
                    st.session_state.last_results = item
                    st.rerun()

# =====================================================================
# 🧾 FOOTER
# =====================================================================

st.markdown("---")
st.caption("Built with ❤️ using Groq + Serper API | Reliable web search with AI-powered analysis")

if __name__ == "__main__":
    main()
