from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import fitz  
import os
from  resumes import resume_links
from jobs import job_desc

model = SentenceTransformer("all-MiniLM-L6-v2")

# Directory to save downloaded PDFs
pdf_dir = "pdf_resumes"

# Create directory if it doesn't exist
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

# Function to download a PDF file from a URL
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


resumes = []

for i, url in enumerate(resume_links):
    pdf_path = os.path.join(pdf_dir, f"resume_{i+1}.pdf")
    download_pdf(url, pdf_path)
    resume_text = extract_text_from_pdf(pdf_path)
    resumes.append(resume_text)


job_descriptions = job_desc


resume_embedding = model.encode(resumes)
job_embedding = model.encode(job_descriptions)



# Calculate similarity
similarity_matrix = cosine_similarity(job_embedding, resume_embedding)

results=[]

top_n=50
top_resume_indices= np.argsort(similarity_matrix, axis=1)[:, top_n:]

for job_index, resume_indice in enumerate(top_resume_indices):
    for rank, resume_index in enumerate(reversed(resume_indice)):
        results.append({
            "Job":job_descriptions[job_index],
            "Resume": resumes[resume_index],
            "Rank": rank + 1
        })





