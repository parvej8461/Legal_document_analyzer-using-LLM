import mlflow
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk import sent_tokenize, word_tokenize
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load SpaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

# Set up MLflow
mlflow.set_experiment("Legal Document Analyzer")

class LegalDocumentAnalyzer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        # Load pre-trained model for inconsistency detection
        self.inconsistency_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.inconsistency_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        
        self.tfidf_vectorizer = TfidfVectorizer()

    def summarize_document(self, document):
        chunks = self.split_into_chunks(document)
        summaries = [self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
        return " ".join(summaries)
    
    def extract_key_points(self, document):
        key_points = []
        questions = [
            "Who are the parties involved in this agreement?",
            "What are the key dates mentioned in the document?",
            "What are the main obligations of each party?",
            "Are there any specific conditions or requirements mentioned?",
            "What are the terms of payment or financial considerations?",
        ]
        for question in questions:
            answer = self.qa_model(question=question, context=document)
            key_points.append(f"{question} {answer['answer']}")
        return key_points
    
    def flag_issues(self, document):
        issues = []
        
        # Vague language detection
        vague_terms = ["may", "might", "could", "subject to change", "reasonable efforts", "as soon as practicable"]
        for term in vague_terms:
            if term in document.lower():
                issues.append(f"Contains potentially vague language: '{term}'")
        
        # Inconsistency detection
        sentences = sent_tokenize(document)
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                inputs = self.inconsistency_tokenizer(sentences[i], sentences[j], return_tensors="pt", truncation=True, padding=True)
                outputs = self.inconsistency_model(**inputs)
                if outputs.logits.argmax().item() == 1:
                    issues.append(f"Potential inconsistency detected between: '{sentences[i]}' and '{sentences[j]}'")
        
        # Missing information detection
        required_sections = ["term", "termination", "confidentiality", "governing law", "dispute resolution"]
        doc_lower = document.lower()
        for section in required_sections:
            if section not in doc_lower:
                issues.append(f"Missing important section: {section}")
        
        # Named Entity Recognition for party identification
        doc = nlp(document)
        parties = set([ent.text for ent in doc.ents if ent.label_ == "ORG"])
        if len(parties) < 2:
            issues.append("Potential issue: Less than two parties identified in the document")
        
        # Detect potential contradictions using semantic similarity
        sentences = sent_tokenize(document)
        sentence_vectors = self.tfidf_vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if similarity_matrix[i][j] > 0.8 and any(word in sentences[i].lower() for word in ["not", "except", "but"]):
                    issues.append(f"Potential contradiction: High similarity but possible negation between '{sentences[i]}' and '{sentences[j]}'")
        
        return issues
    
    def extract_definitions(self, document):
        definitions = {}
        pattern = r'"([^"]+)"\s+means\s+([^.]+)'
        matches = re.findall(pattern, document)
        for match in matches:
            definitions[match[0]] = match[1].strip()
        return definitions
    
    def split_into_chunks(self, document, chunk_size=1000):
        sentences = sent_tokenize(document)
        chunks = []
        current_chunk = []
        current_size = 0
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

# Streamlit UI
st.title("Legal Document Analyzer")

uploaded_file = st.file_uploader("Choose a legal document", type="txt")
if uploaded_file is not None:
    document = uploaded_file.read().decode("utf-8")
    analyzer = LegalDocumentAnalyzer()
    
    with mlflow.start_run():
        summary = analyzer.summarize_document(document)
        key_points = analyzer.extract_key_points(document)
        issues = analyzer.flag_issues(document)
        definitions = analyzer.extract_definitions(document)
        
        st.subheader("Summary")
        st.write(summary)
        
        st.subheader("Key Points")
        for point in key_points:
            st.write(point)
        
        st.subheader("Potential Issues")
        for issue in issues:
            st.write(issue)
        
        st.subheader("Definitions")
        for term, definition in definitions.items():
            st.write(f"**{term}**: {definition}")
        
        # Log results with MLflow
        mlflow.log_param("document_length", len(document))
        mlflow.log_metric("num_key_points", len(key_points))
        mlflow.log_metric("num_issues", len(issues))
        mlflow.log_metric("num_definitions", len(definitions))

        # Log the summary as an artifact
        with open("summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("summary.txt")