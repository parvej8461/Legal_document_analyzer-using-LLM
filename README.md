---

#  Legal Document Analyzer

The **Legal Document Analyzer** is an intelligent NLP-powered tool designed to process and analyze legal documents efficiently. It uses natural language processing techniques to **summarize**, **extract key points**, **identify definitions**, and **flag potential issues** in legal content. With **MLflow** integration, all analyses are logged for reproducibility and experiment tracking.

---

##  Features

- **Summarization**  
  Automatically generates concise summaries of lengthy legal documents.

- **Key Point Extraction**  
  Identifies and highlights critical elements such as parties, key dates, obligations, etc.

- **Issue Detection**  
  Flags ambiguous language, contradictions, missing sections, and insufficient details.

- **Definition Extraction**  
  Automatically extracts and organizes definitions found within the document.

- **MLflow Integration**  
  Tracks and logs metrics, summaries, and parameters to ensure reproducibility and experiment management.

---

##  Technologies Used

- **MLflow** – Experiment tracking and artifact logging  
- **Hugging Face Transformers**  
  - `facebook/bart-large-cnn` for summarization  
  - `deepset/roberta-base-squad2` for question answering  
- **SpaCy** – Named Entity Recognition (NER)  
- **Scikit-learn** – TF-IDF and cosine similarity  
- **Streamlit** – Web UI for document uploads and analysis  
- **Regex** – Legal definition pattern recognition  
- **NLTK** – Sentence tokenization  

---

##  Installation

Follow these steps to set up the Legal Document Analyzer:

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/legal-document-analyzer.git
cd legal-document-analyzer
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the application:**

```bash
streamlit run app.py
```

4. *(Optional)* **Start the MLflow UI:**

```bash
mlflow ui
```

---

##  Usage

1. **Upload a Legal Document**  
   Use the Streamlit interface to upload a `.txt` legal document.

2. **View Results**  
   Once processed, the app will display:
   - 🔹 **Summary** of the document  
   - 🔹 **Key Points** extracted from the content  
   - 🔹 **Potential Issues** like vague or missing clauses  
   - 🔹 **Definitions** found in the document  

3. **Track Results with MLflow**  
   Check the MLflow UI to view experiment logs, including:
   - Parameters
   - Metrics
   - Summarized text and extracted data

---

##  Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository  
2. **Create a feature branch:**

```bash
git checkout -b feature/your-feature
```

3. **Commit your changes:**

```bash
git commit -am 'Add some feature'
```

4. **Push to your branch:**

```bash
git push origin feature/your-feature
```

5. **Open a Pull Request**  

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

