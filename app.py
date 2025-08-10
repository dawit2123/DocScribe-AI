import streamlit as st
import pandas as pd
import spacy
from collections import Counter
import re
from transformers import pipeline, set_seed
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# --- DocScribeAgent Class Definition (Copied from Colab Cell above) ---
class DocScribeAgent:
    def __init__(self, corpus_df_path, nlp_spacy_model_inst, ner_hf_pipeline_inst,
                 abs_summarizer_inst, ext_summarizer_sumy_inst):
        # Load corpus
        self.corpus_df = pd.read_csv(corpus_df_path)
        
        # Assign pre-loaded model instances
        self.nlp_spacy = nlp_spacy_model_inst
        self.ner_hf = ner_hf_pipeline_inst
        self.abs_summarizer = abs_summarizer_inst
        self.ext_summarizer = ext_summarizer_sumy_inst
        
        self.memory = {}
        self.long_term_kb = {}

        self.multi_doc_synthesizer = self.abs_summarizer 

    def _parse_query_and_set_intent(self, query: str) -> dict:
        query_lower = query.lower()
        intent = "UNKNOWN"
        parameters = {}

        if "summarize" in query_lower or "overview" in query_lower or "brief" in query_lower or "summary" in query_lower:
            intent = "SUMMARIZE_TOPIC"
            match = re.search(r'(?:on|about|for|regarding)\s+(.*?)(?:\?|\.$|$)', query_lower)
            parameters['topic'] = match.group(1).strip() if match else "general news"
        elif "who are" in query_lower or "people" in query_lower or "organizations" in query_lower or "involved in" in query_lower:
            intent = "GET_ENTITY_DETAILS"
            match = re.search(r'(?:in|for|involved in)\s+(.*?)(?:\?|\.$|$)', query_lower)
            parameters['scope_topic'] = match.group(1).strip() if match else "recent events"
            if "people" in query_lower or "who are" in query_lower:
                parameters['entity_types'] = ["PERSON"]
            if "organizations" in query_lower:
                parameters['entity_types'] = parameters.get('entity_types', []) + ["ORG"]
            if not parameters.get('entity_types'):
                parameters['entity_types'] = ["PERSON", "ORG", "GPE"]
        elif "financial" in query_lower or "money" in query_lower or "figures" in query_lower or "cost" in query_lower:
            intent = "EXTRACT_FINANCIALS"
            match = re.search(r'(?:of|for|by)\s+(.*?)(?:\?|\.$|$)', query_lower)
            parameters['target'] = match.group(1).strip() if match else ""
        
        parameters['original_query'] = query
        return {"intent": intent, "parameters": parameters}

    def _retrieve_documents_semantic(self, keywords: str, num_docs: int = 5) -> pd.DataFrame:
        if not keywords:
            return self.corpus_df.sample(num_docs, random_state=42)
        
        pattern = r'\b(?:' + '|'.join(map(re.escape, keywords.split())) + r')\b'
        
        filtered_df = self.corpus_df[self.corpus_df['full_text'].str.contains(pattern, case=False, na=False)]
        
        if 'date' in filtered_df.columns:
            filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
            filtered_df = filtered_df.sort_values(by='date', ascending=False)

        if filtered_df.empty:
            general_terms_df = self.corpus_df[self.corpus_df['full_text'].str.contains(keywords.split()[0] if keywords else '', case=False, na=False)]
            return general_terms_df.head(num_docs)
            
        return filtered_df.head(num_docs)

    def _generate_abstractive_summary(self, text: str, max_len: int = 100, min_len: int = 10) -> str:
        if not text or not text.strip(): return ""
        truncated_text = text[:1000] 
        try:
            summary = self.abs_summarizer(truncated_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            return f"Abstractive summary failed: {e}"

    def _extract_all_entities(self, text: str) -> list:
        spacy_ents = self.nlp_spacy(text)
        hf_ents = self.ner_hf(text)

        combined_entities = []
        for ent in spacy_ents.ents:
            combined_entities.append((ent.text, ent.label_, "SpaCy"))
        for ent in hf_ents:
            combined_entities.append((ent['word'], ent['entity_group'], "HF"))
        return combined_entities

    def _extract_key_info_regex(self, text: str) -> dict:
        date_patterns = [r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s+\d{4})?\b', r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', r'\b\d{4}\b']
        money_patterns = [r'\$[\s]*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', r'€[\s]*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', r'\b\d+\s*(?:million|billion|trillion)\b(?:\s+dollars|\s+euros|\s+pounds)?']
        percentage_patterns = [r'\b\d{1,3}(?:\.\d{1,2})?\s*%', r'\b\d{1,3}(?:\.\d{1,2})?\s*percent\b']

        extracted_data = {
            "dates": list(set([m for p in date_patterns for m in re.findall(p, text, re.IGNORECASE)])),
            "money": list(set([m for p in money_patterns for m in re.findall(p, text, re.IGNORECASE)])),
            "percentages": list(set([m for p in percentage_patterns for m in re.findall(p, text, re.IGNORECASE)]))
        }
        return extracted_data

    def process_query(self, query: str) -> str:
        parsed_query = self._parse_query_and_set_intent(query)
        self.memory['last_parsed_query'] = parsed_query 

        intent = parsed_query['intent']
        params = parsed_query['parameters']
        response_parts = [f"🔍 **DocScribe Processing:** _'{query}'_ \n"]

        if intent == "SUMMARIZE_TOPIC":
            topic = params.get('topic', '')
            response_parts.append(f"🎯 **Intent:** Generating concise summaries on _'{topic}'_.")
            relevant_docs = self._retrieve_documents_semantic(topic, num_docs=3)

            if relevant_docs.empty:
                response_parts.append("😓 No relevant documents found. Please try a different topic or broaden your query.")
            else:
                response_parts.append(f"✨ Found **{len(relevant_docs)}** highly relevant articles. Summarizing...\n")
                
                individual_summaries = []
                for i, row in relevant_docs.iterrows():
                    summary = self._generate_abstractive_summary(row['full_text'], max_len=80, min_len=20)
                    response_parts.append(f"\n--- Article: **{row['headline']}** (Category: {row['category']}) ---\n")
                    response_parts.append(f"📝 **Summary:** {summary}\n")
                    individual_summaries.append(summary)

                if len(individual_summaries) > 1:
                    combined_text = " ".join(individual_summaries)
                    if len(combined_text.split()) > 500:
                        combined_text = " ".join(combined_text.split()[:500]) + "..."
                    synthesized_summary = self.multi_doc_synthesizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                    response_parts.append("\n--- **Cross-Document Synthesis by DocScribe** ---\n")
                    response_parts.append(f"✨ **Unified Overview:** {synthesized_summary}\n")
                    self.memory['last_synthesized_summary'] = synthesized_summary

        elif intent == "GET_ENTITY_DETAILS":
            topic = params.get('scope_topic', '')
            entity_types = params.get('entity_types', ["PERSON", "ORG", "GPE"])
            response_parts.append(f"🎯 **Intent:** Identifying key {', '.join(entity_types)} entities in news regarding _'{topic}'_.")
            relevant_docs = self._retrieve_documents_semantic(topic, num_docs=7)

            if relevant_docs.empty:
                response_parts.append("😓 No relevant documents found. Perhaps try a broader scope?")
            else:
                response_parts.append(f"✨ Found **{len(relevant_docs)}** relevant articles. Extracting entities...\n")
                aggregated_entities = Counter()
                
                for _, row in relevant_docs.iterrows():
                    all_ents = self._extract_all_entities(row['full_text'])
                    for text, label, source in all_ents:
                        if label in entity_types:
                            if label == "PERSON":
                                aggregated_entities[(text.lower(), label)] += 1
                            else: 
                                aggregated_entities[(text, label)] += 1
                
                response_parts.append("\n--- **Top Entities Discovered** ---\n")
                if aggregated_entities:
                    sorted_entities = sorted(aggregated_entities.items(), key=lambda item: (-item[1], item[0][0]))
                    for (entity_text, entity_label), count in sorted_entities[:10]:
                        response_parts.append(f"💡 **{entity_text.title()}** ({entity_label}): {count} mentions\n")
                else:
                    response_parts.append("🤷‍♀️ No specific entities of the requested types found in these documents.")
                self.memory['last_extracted_entities'] = aggregated_entities

        elif intent == "EXTRACT_FINANCIALS":
            target = params.get('target', '')
            response_parts.append(f"🎯 **Intent:** Extracting financial insights related to _'{target}'_.")
            relevant_docs = self._retrieve_documents_semantic(target, num_docs=5)

            if relevant_docs.empty:
                response_parts.append("😓 No documents found mentioning financial information for this target.")
            else:
                response_parts.append(f"✨ Found **{len(relevant_docs)}** relevant articles. Extracting financial figures and dates...\n")
                extracted_financial_data = []
                for _, row in relevant_docs.iterrows():
                    info = self._extract_key_info_regex(row['full_text'])
                    if info['money'] or info['percentages'] or info['dates']:
                        extracted_financial_data.append({
                            "headline": row['headline'],
                            "money": info['money'],
                            "percentages": info['percentages'],
                            "dates": info['dates']
                        })
                
                if extracted_financial_data:
                    for item in extracted_financial_data:
                        response_parts.append(f"- **Article:** _'{item['headline']}'_\n")
                        response_parts.append(f"  💰 Money: {', '.join(item['money']) if item['money'] else 'N/A'}\n")
                        response_parts.append(f"  📈 Percentages: {', '.join(item['percentages']) if item['percentages'] else 'N/A'}\n")
                        response_parts.append(f"  📅 Dates: {', '.join(item['dates']) if item['dates'] else 'N/A'}\n")
                        response_parts.append("--- \n")
                else:
                    response_parts.append("🤷‍♀️ No specific financial figures, percentages, or dates found in these articles.")
                self.memory['last_extracted_financials'] = extracted_financial_data

        else: # UNKNOWN Intent
            response_parts.append("🤔 **DocScribe is pondering...**")
            response_parts.append("I couldn't quite grasp that query. Could you try phrasing it differently?")
            response_parts.append("For example:\n- _'Summarize news on AI in healthcare'_")
            response_parts.append("- _'Who are the key people and organizations involved in the space industry?'_")
            response_parts.append("- _'What are the financial figures for Tesla?'_")
        
        return "\n".join(response_parts)

# --- Streamlit UI Code ---

st.set_page_config(
    page_title="DocScribe: Intelligent Document Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "cool and amusing" UI
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6; /* Light gray background */
}
.reportview-container {
    background: #f0f2f6;
}
.sidebar .sidebar-content {
    background-color: #e0e5ed; /* Slightly darker sidebar */
}
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50; /* Darker blue-gray for headers */
}
.stMarkdown {
    color: #34495e; /* Standard text color */
}
.stTextInput > div > div > input {
    border: 2px solid #3498db; /* Blue border for input */
    border-radius: 8px;
    padding: 10px;
}
.stButton > button {
    background-color: #3498db; /* Blue button */
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    background-color: #2980b9; /* Darker blue on hover */
    transform: translateY(-2px);
}
.stAlert {
    border-radius: 8px;
    background-color: #ecf0f1; /* Light gray for alerts */
    border: 1px solid #bdc3c7;
}
.css-1r6dmym { /* Target markdown block styling for response */
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.animated-spin {
    animation: spin 2s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)


# --- Global Model Loading (using st.cache_resource) ---
# This ensures models are loaded only once across Streamlit app runs
@st.cache_resource
def load_global_models():
    with st.spinner("🧠 Booting up DocScribe's AI Brain... This might take a moment!"):
        spacy.cli.download("en_core_web_sm") # Ensure spacy model is downloaded for deployment
        nlp_spacy = spacy.load("en_core_web_sm")
        
        ner_hf = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=0) 
        abs_summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=0)

        # Sumy models (NLTK data needs to be present for these too)
        # NLTK data downloads are usually handled by the !pip install / nltk.download commands in Colab directly
        # If running locally without prior NLTK setup, add:
        # import nltk
        # nltk.download('punkt', quiet=True)
        # nltk.download('stopwords', quiet=True)
        # nltk.download('wordnet', quiet=True)
        # nltk.download('omw-1.4', quiet=True)
        stemmer_sumy_inst = Stemmer("english")
        summarizer_textrank_sumy_inst = TextRankSummarizer(stemmer_sumy_inst)
        summarizer_textrank_sumy_inst.stop_words = get_stop_words("english")
        
    return nlp_spacy, ner_hf, abs_summarizer, summarizer_textrank_sumy_inst

# Load models and initialize agent
nlp_spacy_model_inst, ner_hf_pipeline_inst, abs_summarizer_inst, summarizer_textrank_sumy_inst = load_global_models()

# Ensure the processed data file exists. This path must be absolute or relative to where app.py runs.
# In Colab, it will be '/content/news_data/df_for_part3_sample.csv'
corpus_path = "./Assets/df_for_part3_sample.csv"
if not os.path.exists(corpus_path):
    st.error(f"DocScribe's data brain is missing! Could not find {corpus_path}. Please ensure Part 1 and Part 2 cells are run to generate the data file.")
    st.stop() # Stop the app if data is not found

# Initialize DocScribe Agent
if 'docscribe_agent' not in st.session_state:
    st.session_state.docscribe_agent = DocScribeAgent(
        corpus_df_path=corpus_path,
        nlp_spacy_model_inst=nlp_spacy_model_inst,
        ner_hf_pipeline_inst=ner_hf_pipeline_inst,
        abs_summarizer_inst=abs_summarizer_inst,
        ext_summarizer_sumy_inst=summarizer_textrank_sumy_inst
    )
    st.session_state.chat_history = []


# --- UI Elements ---
st.title("🤖 DocScribe: Your Intelligent Document Analysis Agent")
st.markdown("##### _Unveiling insights from the vast ocean of text!_")

st.sidebar.header("About DocScribe's Brain")
st.sidebar.markdown(
    """
    DocScribe is an advanced AI agent designed to distill complex information from unstructured text documents.
    It combines cutting-edge Natural Language Processing (NLP) techniques for:
    - **Intelligent Query Understanding**: Deciphers your requests.
    - **Semantic Document Retrieval**: Finds the most relevant articles.
    - **Multi-faceted Information Extraction**: Pinpoints key entities, dates, and financials.
    - **Advanced Abstractive Summarization**: Creates fluent, concise summaries, even across multiple documents.
    """
)
st.sidebar.image("https://em-content.zobj.net/source/microsoft/319/robot_1f916.png", width=100) # Fun icon
st.sidebar.markdown("---")
st.sidebar.header("How to Talk to DocScribe")
st.sidebar.markdown(
    """
    Try asking DocScribe things like:
    - `Summarize recent news on renewable energy`
    - `Who are the key people and organizations involved in climate change talks?`
    - `What are the financial figures for Google?`
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, Streamlit, NLTK, SpaCy, & Hugging Face Transformers.")


# Main chat interface
query = st.text_input("Ask DocScribe a question:", key="user_query", help="E.g., Summarize news on AI advancements")

if st.button("Unveil Insights! 🚀"):
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.spinner("DocScribe is meticulously analyzing... 🧐"):
            # Call the agent to process the query
            agent_response = st.session_state.docscribe_agent.process_query(query)
            st.session_state.chat_history.append({"role": "agent", "content": agent_response})
    else:
        st.warning("Please enter a query for DocScribe to analyze!")

# Display chat history
st.markdown("---")
st.subheader("DocScribe's Analysis:")

for message in reversed(st.session_state.chat_history): # Show most recent at top
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**DocScribe:**\n{message['content']}")
    st.markdown("---")
