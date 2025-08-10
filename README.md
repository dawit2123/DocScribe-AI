# 🤖 DocScribe: Your Intelligent Document Analysis Agent

<p align="center">
  <img src="https://em-content.zobj.net/source/microsoft/319/robot_1f916.png" width="150">
</p>

---

## ✨ **Welcome, Curious Mind!**

Ever felt like you're drowning in a sea of documents, desperately trying to find that one golden nugget of insight? Or perhaps you just wish someone (or something!) could summarize all the latest news on [insert overwhelming topic here] for you?

**Fear not!** Presenting **DocScribe**, your highly-caffeinated, meticulously organized, and surprisingly witty AI assistant for conquering the world of unstructured text! We built DocScribe not just to work, but to *impress* – because who says advanced NLP can't be fun?

---

## 🚀 **What's This Magic All About? (Project Objective)**

DocScribe isn't your average text-muncher. It's an intelligent agent designed to tackle real-world information overload by:

1.  **🔍 Extracting the Good Stuff:** Pulling out key entities, facts, and figures from mountains of text.
2.  **📝 Summarizing Like a Pro:** Condensing lengthy articles into crisp, digestible insights.
3.  **🧠 Reasoning Smartly:** Chaining its NLP superpowers to answer complex questions and synthesize information across multiple documents.

Think of it as a personal intelligence analyst, but without the coffee stains or the need for a desk!

---
## Demo
![preview 1](https://github.com/dawit2123/DocScribe-AI/blob/main/Assets/preview1.png)
![Demo](https://github.com/dawit2123/DocScribe-AI/blob/main/Assets/System%20Design.png)
![preview 2](https://github.com/dawit2123/DocScribe-AI/blob/main/Assets/preview2.png)
# Design Image
![Design](https://github.com/dawit2123/DocScribe-AI/blob/main/Assets/demo.png)
# Brief Report
[Brief Report](https://github.com/dawit2123/DocScribe-AI/blob/main/DocScribe%20AI%20Brief%20report.pdf)

## 🛠️ **Getting DocScribe Up and Running (Setup Instructions)**

Ready to unleash DocScribe's brilliance? Follow these simple steps (preferably in a Google Colab environment for that sweet, sweet GPU acceleration!):

1.  **Clone this Repository (if applicable):**
    ```bash
    git clone https://github.com/dawit2123/DocScribe-AI
    cd DocScribe-AI
    ```
2.  **Fire Up Google Colab:** Open your `.ipynb` notebook in Google Colab.
3.  **Install the Essentials:** Run the initial setup cells (usually near the top) to install all required libraries. DocScribe is quite the polyglot, needing a few friends like `pandas`, `nltk`, `spacy`, `transformers`, `sumy`, `streamlit`, and `pyngrok`.
    *   **NLTK Data:** The setup script will automatically download necessary NLTK data (`punkt`, `stopwords`, `wordnet`).
    *   **SpaCy Model:** It also downloads `en_core_web_sm` – DocScribe needs its foundational brain!
4.  **Kaggle API Key (For Dataset Download):**
    *   **Get Your Key:** Head over to your Kaggle account settings (`kaggle.com/<your_username>/account`). Under the 'API' section, click 'Create New API Token' to download `kaggle.json`.
    *   **Upload to Colab:** In your Colab session, upload this `kaggle.json` file directly to the `/content/` directory (look for the folder icon on the left sidebar).
    *   **Secure It:** The setup script will handle making it secure (`!chmod 600 ~/.kaggle/kaggle.json`).
5.  **Ngrok Authentication (For Live Demo!):**
    *   **Get Your Token:** Sign up at [ngrok.com](https://ngrok.com/) and grab your authtoken from your dashboard.
    *   **Plug It In:** In the Colab cell dedicated to `ngrok` setup, **replace `"YOUR_NGROK_AUTH_TOKEN"` with your actual token.** DocScribe can't go live without it!

---

## 📚 **DocScribe's Knowledge Base (Dataset Source & Preprocessing)**

DocScribe is trained on the juicy bits of the **Kaggle News Category Dataset** (available [here](https://www.kaggle.com/datasets/rmisra/news-category-dataset)). This fantastic corpus provides news headlines and short descriptions, perfect for our geopolitical analyst scenario.

Before DocScribe gets its hands (or rather, algorithms) dirty, the raw text undergoes a rigorous spa-day, also known as **Preprocessing**:

*   **Combined Brilliance:** Headlines and short descriptions are fused into a single `full_text` field – because two sources are better than one!
*   **Case-Sensitivity? Never Heard of It!** Everything is **lowercased** for consistency.
*   **Token Tango:** Text is broken down into individual `tokens` (words).
*   **Stop-Word Purge:** Common, less meaningful words (like "the", "is", "a") are meticulously removed. DocScribe likes to get straight to the point!
*   **Linguistic Facelift (Lemmatization):** Words are reduced to their base dictionary form (e.g., "running," "ran," "runs" all become "run"). This ensures DocScribe understands the core meaning.

This meticulous cleaning ensures DocScribe's insights are sharp, accurate, and free from linguistic clutter!

---

## 🎬 **DocScribe in Action! (How to Run Each Module)**

Our Colab notebook is structured to walk you through DocScribe's development, step by step. Simply **"Run all cells"** in your Google Colab environment. Each section builds upon the last:

*   **Part 1: Data Preparation & Exploration:** Witness the data loading, preprocessing, and exploratory analysis (document length, most frequent words, initial entity frequency). It's where DocScribe first learns to crawl through text!
*   **Part 2: Information Extraction & Summarization:** See DocScribe flex its NLP muscles!
    *   **Rule-Based Extraction:** Watch it sniff out dates, money, and percentages with regex!
    *   **NER with SpaCy:** Observe its core entity-identifying superpowers (persons, orgs, locations).
    *   **Transformer-Based NER (Hugging Face):** Experience the bleeding-edge entity recognition that brings true detail.
    *   **Extractive Summarization (TextRank):** See it pick out the most important sentences.
    *   **Abstractive Summarization (T5-small):** Marvel as it paraphrases and synthesizes human-like summaries!
*   **Part 3: DocScribe Agent Design & Live Demo:** This is the grand finale!
    *   The notebook defines the `DocScribeAgent` class and prepares everything.
    *   The `app.py` script (which you'll find in your Colab files after running the "Create app.py" cell) powers the Streamlit interface.
    *   The final cells will launch Streamlit and provide an `ngrok` public URL. Click that link, and **BAM!** You're interacting with DocScribe live in your browser!

---

## 🧠 **Inside DocScribe's Brilliant Mind (Agent Design Explanation)**

DocScribe isn't just a collection of NLP models; it's a meticulously designed intelligent agent.

**Scenario:** DocScribe operates within a geopolitical think tank, where analysts are overwhelmed by the sheer volume of global news. DocScribe's mission: to deliver precise, synthesized intelligence on demand.

**Agent's Goal:** To be the ultimate information assistant:
1.  **Rapid Comprehension:** Provide concise, relevant summaries.
2.  **Precise Intelligence:** Accurately extract key entities and data.
3.  **Cross-Document Synthesis:** Answer complex queries by intelligently combining insights from multiple sources.
4.  **Proactive Monitoring:** Surface emerging trends.

**DocScribe's Tools (Its Superpowers!):**
DocScribe's core capabilities (from Part 2) are its specialized tools:

*   **Document Retrieval Tool:** DocScribe's "information bloodhound," using **semantic search** (conceptualized via vector embeddings) to fetch the most relevant articles.
*   **Named Entity Recognition (NER) Tool:** DocScribe's "digital detective," combining **SpaCy** and **Hugging Face Transformers** to find every crucial 'who,' 'what,' and 'where.'
*   **Key Information Extraction (KIE) Tool:** DocScribe's "data accountant," precisely extracting numbers (money, percentages) and dates using **regex patterns**.
*   **Abstractive Summarization Tool:** DocScribe's "narrative artist," generating fluent, new summaries with **Hugging Face T5-small**.
*   **Extractive Summarization Tool:** DocScribe's "quote curator," picking out direct, salient sentences using **TextRank**.

**Reasoning/Planning Strategy (How DocScribe Thinks!):**
DocScribe doesn't just act; it plans!

1.  **Query Parsing & Intent Recognition:** It first deciphers your query's intent (e.g., "Summarize," "Extract Entities").
2.  **Intelligent Document Selection:** It then uses its Retrieval Tool to fetch *exactly* the right documents.
3.  **Dynamic Tool Chaining & Cross-Document Synthesis:** This is DocScribe's genius!
    *   **For Summaries:** It runs individual articles through the **Abstractive Summarizer**, then takes those summaries and feeds them into a *second pass* of the same summarizer (acting as a **Multi-Document Synthesizer**) to create a single, unified narrative across all sources.
    *   **For Entity Lists:** It dispatches both **NER tools** and the **KIE tool** to find all actors and dates, then intelligently aggregates and prioritizes them.
    *   **For Financial Data:** It targets the **KIE tool** specifically for monetary values and percentages, presenting them with context.

**Memory Component (DocScribe Never Forgets!):**
DocScribe boasts a sophisticated memory architecture for next-level interaction:

*   **Short-Term Contextual Memory:** Like a highly attentive assistant, it remembers your current conversation thread (`Streamlit's session_state`) for fluid follow-up questions.
*   **Long-Term Knowledge Base / Semantic Cache:** This is DocScribe's ever-growing "brain library" – a persistent store of processed data, document embeddings, extracted entities, and cached summaries. This enables faster retrieval, consistent responses, and the potential for advanced **trend analysis** by remembering what it's learned over time!

---


## 🌐 **Experience DocScribe Live!**

Once the Streamlit & Ngrok cells finish running in Colab, you'll see a public URL. Click it, and interact with DocScribe directly! Ask it about "renewable energy," "key people in space exploration," or "financial figures for Tesla." Prepare to be amazed!

<p align="center">
  *DocScribe: Cutting through the noise, one insight at a time!*

</p>



