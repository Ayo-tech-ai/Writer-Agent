import streamlit as st
import requests
import json
from io import BytesIO

# =========================
# 🌍 APP CONFIGURATION
# =========================
st.set_page_config(
    page_title="Groq-Powered Agentic Researcher",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 🎨 CUSTOM STYLING
# =========================
st.markdown("""
    <style>
        /* General app styling */
        body {
            font-family: 'Segoe UI', sans-serif;
        }

        .main {
            padding: 1rem 2rem;
        }

        /* Section titles */
        h1, h2, h3 {
            color: #222;
        }

        /* Card-like look for outputs */
        .stMarkdown {
            background-color: #f9fafb;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }

        /* Responsive fix for desktop */
        @media (min-width: 1024px) {
            .main > div {
                max-width: 1000px;
                margin: auto;
            }
        }

        /* Make text more readable on phone */
        @media (max-width: 768px) {
            .main {
                padding: 0.8rem;
            }
            h1 { font-size: 1.6rem; }
            h2 { font-size: 1.3rem; }
        }

        /* Download button look */
        .stDownloadButton button {
            border-radius: 8px;
            background-color: #0072ff;
            color: white;
            border: none;
            padding: 0.6rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🧠 HELPER FUNCTION – GROQ LLM CALL
# =========================
def call_groq_llm(prompt, system_message, api_key, model="mixtral-8x7b"):
    """Calls the Groq API for text generation."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"Groq API Error: {response.text}")
        return None


# =========================
# 🔍 HELPER FUNCTION – SERPER SEARCH
# =========================
def search_serper(query, api_key, max_results=5):
    """Searches the web using Serper.dev (Google Search API)."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query})

    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        data = response.json()
        results = data.get("organic", [])[:max_results]
        return results
    else:
        st.error("❌ Serper API error. Check your key.")
        return []


# =========================
# ⚙️ RESEARCH WORKFLOW
# =========================
def execute_research_workflow(query, groq_api_key, serper_api_key, model, max_results):
    # Step 1 – Web Search
    with st.status("🔎 Performing intelligent web search...", expanded=True) as status:
        search_results = search_serper(query, serper_api_key, max_results)
        if not search_results:
            st.warning("No results found.")
            return None
        status.update(label="✅ Web search completed", state="complete")

    # Step 2 – Research Report
    with st.status("🧠 Generating research report...", expanded=True) as status:
        search_text = "\n".join(
            [f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}\nLink: {r.get('link')}" for r in search_results]
        )

        research_prompt = f"""
        You are an AI researcher analyzing recent findings about the topic below.

        TOPIC: {query}

        WEB RESULTS:
        {search_text}

        Write a structured research summary that highlights:
        - Key developments, facts, and statistics
        - Relevant Nigerian or African context (if applicable)
        - Challenges, opportunities, and insights
        - Short conclusion (2–3 sentences)
        """

        research_report = call_groq_llm(
            research_prompt,
            "You are a professional research analyst. Write structured, factual, and insightful reports.",
            groq_api_key,
            model
        )
        status.update(label="✅ Research report completed", state="complete")

    # Step 3 – LinkedIn Post
    with st.status("💬 Creating LinkedIn post...", expanded=True) as status:
        linkedin_post_prompt = f"""
        Based on the following research report, create a high-quality LinkedIn post that summarizes and humanizes the key findings.

        RESEARCH REPORT:
        {research_report}

        Write a professional, engaging LinkedIn post that:
        - Has a catchy headline with an emoji (🚀, 💡, 🌍, etc.)
        - Begins with a strong hook that sparks curiosity
        - Contains 3–6 short sections with emoji headers (e.g., 🌾 The New Farming Frontier)
        - Clearly explains insights, data, and implications in conversational tone
        - Feels authentic — not robotic or academic
        - Ends with a reflective question or call to action
        - Includes 5–8 relevant hashtags
        - Optionally uses emojis for visual flow
        - Length: 1000–1500 words
        """

        linkedin_post = call_groq_llm(
            linkedin_post_prompt,
            "You are a professional LinkedIn storyteller who writes engaging, research-based posts that educate and inspire professionals.",
            groq_api_key,
            model
        )

        status.update(label="✅ LinkedIn post created successfully", state="complete")

    # ✅ Return all outputs
    return {
        "research_report": research_report,
        "linkedin_post": linkedin_post,
        "search_results": search_results
    }


# =========================
# 🖥️ MAIN APP LAYOUT
# =========================
st.title("🌍 Groq-Powered Agentic Researcher")
st.markdown("#### 🤖 Perform intelligent web research + auto-generate a LinkedIn post")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    serper_key = st.text_input("Serper API Key (Required)", type="password")

    st.markdown("---")
    model = st.selectbox("Select Model", ["mixtral-8x7b", "llama3-70b-8192", "gemma-7b-it"])
    creativity = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05)
    max_results = st.slider("Max Results", 1, 10, 5)
    st.markdown("---")
    st.markdown("💡 Using Serper API for reliable web search")

# =========================
# 🚀 MAIN CONTENT
# =========================
query = st.text_input("🔍 Enter your research topic:", placeholder="e.g., AI in Nigeria")

if st.button("🚀 Start Research", use_container_width=True):
    if not serper_key or not groq_key:
        st.error("❌ Please provide both Groq and Serper API keys.")
    elif not query.strip():
        st.warning("Please enter a research topic.")
    else:
        result = execute_research_workflow(query, groq_key, serper_key, model, max_results)

        if result:
            st.divider()
            st.subheader("📊 Research Report")
            st.markdown(result["research_report"])

            st.divider()
            st.subheader("💼 Generated LinkedIn Post")
            st.markdown(result["linkedin_post"])

            # ✅ Download button
            download_text = result["linkedin_post"]
            download_bytes = BytesIO(download_text.encode('utf-8'))

            st.download_button(
                label="⬇️ Download LinkedIn Post (.txt)",
                data=download_bytes,
                file_name=f"{query.replace(' ', '_')}_LinkedIn_Post.txt",
                mime="text/plain",
                use_container_width=True
            )

            st.success("✅ Research and LinkedIn post generation complete!")

# =========================
# ⚡ FOOTER
# =========================
st.markdown("---")
st.caption("Built with ❤️ using Groq + Serper API | Reliable web research & storytelling assistant.")
