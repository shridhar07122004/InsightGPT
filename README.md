## InsightGPT — AI-Powered News Research Tool

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Configuration](#configuration)
* [Contributing](#contributing)
* [License](#license)

## Overview

**InsightGPT — AI News Research Tool** is a web-based application built using Streamlit. It helps users extract, analyze, and research news articles using HuggingFace embeddings and FAISS for vector indexing. The tool allows users to process multiple URLs and ask questions based on the extracted content with AI-generated responses powered by Groq LLaMA-3 LLM.

## Features

* Supports multiple news article URLs for content extraction.
* Uses HuggingFace Sentence Transformer embeddings for semantic understanding.
* Implements FAISS (Facebook AI Similarity Search) for fast vector retrieval.
* Ask questions and get context-based AI answers.
* Simple and interactive Streamlit UI.

## Installation

### Prerequisites

Before running this tool, ensure you have:

* Python 3.8 or above
* Streamlit installed
* FAISS installed
* Groq API key for LLaMA inference

### Step-by-Step Installation

1. Clone the repository:
   git clone [https://github.com/shridhar07122004/InsightGPT.git](https://github.com/shridhar07122004/InsightGPT.git)
   cd InsightGPT

2. Install dependencies:
   pip install -r requirements.txt

3. Install NLTK data:
   python -c "import nltk; nltk.download('punkt')"

4. Set your Groq API Key:
   export GROQ_API_KEY='your-groq-api-key'

## Usage

1. Navigate to the project folder and run:
   streamlit run main.py

2. Enter up to 3 news URLs in the sidebar.

3. Click **Process URLs** to extract and index data.

4. Type your question, and InsightGPT will provide answers based on the extracted content.

## Dependencies

* Streamlit — UI framework
* LangChain — Text processing and chaining
* FAISS — Vector similarity search
* HuggingFace Sentence Transformers — Embeddings
* Groq LLaMA-3 — LLM for answering queries
* NLTK — Tokenization and text preprocessing

You can find all dependencies in the requirements.txt file.

## Configuration

Set your Groq API key as an environment variable:
export GROQ_API_KEY='your-groq-api-key'

## Contributing

Contributions are welcome. Fork this repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License.

---


