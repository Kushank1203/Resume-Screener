import os
import re
import json
import PyPDF2
import docx
import spacy
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ====================== Setup ======================

# Load spaCy model (use 'en_core_web_sm' if large model is not available)
nlp = spacy.load("en_core_web_lg")

# Load Transformer model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Flask Configuration
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for Flask session
CORS(app)  # Enable CORS for frontend-backend communication

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====================== Resume Parsing ======================

def extract_text_from_resume(file_path):
    """Extract text from PDF or DOCX files."""
    try:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

# ====================== NLP Processing ======================

def extract_skills(text):
    """Extract skills using NLP."""
    doc = nlp(text.lower())
    skills = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            skills.add(token.text)
    return list(skills)

def extract_experience(text):
    """Extract years of experience from resume text."""
    exp_years = re.findall(r'(\d+)\s+years', text.lower())
    if exp_years:
        total_years = sum(map(int, exp_years))
    else:
        total_years = 0
    return total_years

def extract_education(text):
    """Extract degree and field from resume text."""
    degree_pattern = re.compile(r"(bachelor|master|phd|doctorate|associate)\s+(of|in)?\s*([a-zA-Z\s]+)", re.IGNORECASE)
    matches = degree_pattern.findall(text)
    education = []
    for match in matches:
        degree = match[0]
        field = match[2].strip()
        education.append({"degree": degree.title(), "field": field})
    return education

# ====================== Scoring ======================

def score_resume(job_description, resume_text):
    """Score a resume based on job description using TF-IDF similarity."""
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([job_description, resume_text])
        similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    except Exception as e:
        similarity_score = 0
    
    # Skills Matching Score
    job_skills = set(extract_skills(job_description))
    resume_skills = set(extract_skills(resume_text))
    if job_skills:
        skill_match_score = len(job_skills.intersection(resume_skills)) / len(job_skills) * 100
    else:
        skill_match_score = 0

    # Experience Match Score
    job_exp = extract_experience(job_description)
    resume_exp = extract_experience(resume_text)
    if job_exp:
        exp_match_score = min((resume_exp / job_exp) * 100, 100)
    else:
        exp_match_score = 0
    
    # Education Match Score
    job_edu = extract_education(job_description)
    resume_edu = extract_education(resume_text)
    if job_edu and resume_edu:
        edu_match_score = 100
    else:
        edu_match_score = 0

    # Final weighted score
    final_score = (similarity_score * 0.4) + (skill_match_score * 0.3) + (exp_match_score * 0.2) + (edu_match_score * 0.1)

    return {
        'final_score': final_score,
        'skills_match': skill_match_score,
        'exp_match': exp_match_score,
        'edu_match': edu_match_score,
        'matching_skills': list(job_skills.intersection(resume_skills)),
        'missing_skills': list(job_skills.difference(resume_skills))
    }

# ====================== Flask Routes ======================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'job_description' not in request.files or 'resumes' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    job_description_file = request.files['job_description']
    resume_files = request.files.getlist('resumes')

    if not job_description_file or not resume_files:
        return jsonify({'error': 'Invalid file upload'}), 400

    job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(job_description_file.filename))
    job_description_file.save(job_description_path)
    job_description_text = extract_text_from_resume(job_description_path)

    results = []
    for resume in resume_files:
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume.filename))
        resume.save(resume_path)
        resume_text = extract_text_from_resume(resume_path)

        if not resume_text:
            continue
        
        score = score_resume(job_description_text, resume_text)
        results.append({
            'candidate_name': Path(resume.filename).stem,
            'score': score['final_score'],
            'score_details': score,
            'experience': {'total_years': extract_experience(resume_text)},
            'education': extract_education(resume_text)
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    session['rankings'] = results
    return jsonify({'rankings': results})  # ✅ Return JSON instead of redirect

@app.route('/results')
def show_results():
    rankings = session.get('rankings', [])
    return render_template('results.html', rankings=rankings)

# ====================== Run Flask Server ======================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
