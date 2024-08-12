from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Ensure this directory exists

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ''

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resume')

        if not job_description or len(resume_files) < 5:
            return render_template('home.html', message='Please upload at least 5 resumes and provide a job description')

        resumes = []
        resume_paths = []
        for resume_file in resume_files:
            if resume_file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(file_path)
                resume_paths.append(file_path)
                resumes.append(extract_text(file_path))

        # Process resumes and job_description
        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([job_description] + resumes)
        
        job_description_vector = vectors[0]
        resumes_vectors = vectors[1:]

        # Compute cosine similarities
        similarities = cosine_similarity(job_description_vector, resumes_vectors).flatten()

        # Get top 3 resumes and their similarity scores
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_scores = similarities[top_indices]
        top_resumes = [resume_paths[i] for i in top_indices]

        # Combine top resumes and their scores
        top_results = list(zip(top_resumes, top_scores))

        return render_template('home.html', message='Resumes successfully uploaded and processed', top_results=top_results)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
