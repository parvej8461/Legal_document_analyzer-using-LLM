#Legal Document Analyzer
The Legal Document Analyzer is an intelligent tool that helps users process and analyze legal documents. It leverages natural language processing (NLP) techniques to summarize, extract key points, flag potential issues, and identify definitions within legal documents. Additionally, the tool uses MLflow to track and log various metrics and results, enabling experiment tracking and reproducibility.

#Features
Summarization: Automatically summarizes lengthy legal documents into concise summaries.
Key Point Extraction: Extracts crucial information such as parties involved, key dates, obligations, and more.
Issue Detection: Identifies potential problems like vague language, inconsistencies, missing sections, contradictions, and insufficient party identification.
Definition Extraction: Detects and extracts definitions from legal documents.
MLflow Integration: Logs key metrics, parameters, and artifacts (like summaries) for tracking experiments.
Technologies Used
MLflow: For experiment tracking and logging.
Hugging Face Transformers: Used for summarization (facebook/bart-large-cnn) and question answering (deepset/roberta-base-squad2).
SpaCy: For Named Entity Recognition (NER).
Scikit-learn: For TF-IDF vectorization and cosine similarity computation.
Streamlit: Provides an intuitive web UI for users to upload and analyze legal documents.
Regex: For extracting definitions.
NLTK: Used for sentence tokenization.
Installation
To set up the Legal Document Analyzer, follow these steps:

Clone the repository:


git clone https://github.com/your-username/legal-document-analyzer.git
cd legal-document-analyzer
Install dependencies:


pip install -r requirements.txt
Run the application:

streamlit run app.py
(Optional) Set up MLflow tracking:


mlflow ui
Usage
Upload a Legal Document: Once the application is running, upload a legal document (in .txt format) using the file uploader.
View Results: The app will automatically process the document and display the following:
Summary: A concise summary of the document.
Key Points: Extracted key points related to the legal content.
Potential Issues: Detected potential issues such as vague language, missing sections, and contradictions.
Definitions: Extracted legal definitions found in the document.
Track Results with MLflow: The app logs the document analysis results as an MLflow experiment. You can view parameters, metrics, and artifacts (e.g., summary) in the MLflow UI.


#Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -am 'Add some feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.
