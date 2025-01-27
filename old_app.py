from flask import Flask, request, render_template
import re
import os
import glob
import PyPDF2
import docx2txt
from nltk import pos_tag
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk_data_path = "Resume Screening/nltk_data"
nltk.data.path.append(nltk_data_path)

#run below 3 line only first time...
# nltk.download('punkt',download_dir=nltk_data_path)
# nltk.download('wordnet',download_dir=nltk_data_path)
# nltk.download('stopwords',download_dir=nltk_data_path)
# nltk.download('averaged_perceptron_tagger_eng',download_dir=nltk_data_path)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


def email_extract(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails


def data_preprocessing(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove special character and number
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'https?://\s+|www\.\s+', '', text)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(text)

    for sentence in sentences:
        tokens = word_tokenize(sentence)

    skills = []
    tagged_words = pos_tag(tokens)

    # Step 3: Extract potential skills based on POS tagging
    for word, tag in tagged_words:
        # Look for nouns (NN, NNS) or adjectives (JJ, JJS)
        if tag in ['NN', 'NNS', 'JJ', 'JJS']:
            skills.append(word)
    tokens = [lemmatizer.lemmatize(word) for word in skills if word not in stop_words]

    text = ' '.join(tokens)
    return text


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return email_extract(text),data_preprocessing(text)


def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return email_extract(text),data_preprocessing(text)


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return email_extract(text),data_preprocessing(text)


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/user')
def user():
    return render_template('student.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/matching')
def matching():
    return render_template('matching.html')


@app.route('/match', methods=['POST'])
def match():
    if request.method == "POST":
        job_description = request.form['jobDescription']
        resume_files = request.files.getlist("resume")
        top_resume_no = int(request.form["resume_no"])
        if len(resume_files) < top_resume_no:
            return render_template("index.html", message="Please uploads more resumes")
        else:
            resumes = []
            emails = []
            for resume_file in resume_files:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(filename)
                email, resume_text = extract_text(filename)
                resumes.append(resume_text)
                emails.append(email)

            job_description = data_preprocessing(job_description)

            vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
            vectors = vectorizer.toarray()

            job_vector = vectors[0]
            resume_vector = vectors[1:]

            # similarity = cosine_similarity([job_vector], resume_vector)[0]
            #
            # results = []
            # top_indices = similarity.argsort()[::-1]
            # top_resume = [resume_files[i].filename for i in top_indices]
            # similarity_score = [round((similarity[i]) * 100, 2) for i in top_indices]
            # top_email = [str(emails[i]).strip("['']") for i in top_indices]
            #
            # for name, score, email in zip(top_resume, similarity_score, top_email):
            #     results.append({"resume_name": name, "score": score, "email": email})
            #
            # results = results[:top_resume_no]

            #KNN model to find closest resumes
            knn = NearestNeighbors(n_neighbors=top_resume_no, metric='cosine')
            knn.fit(resume_vector)  # Fit on resumes only
            distances, indices = knn.kneighbors([job_vector])

            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                similarity_score = ((1-distances[0][i]) * 100)+35  # Convert distance to similarity percentage
                results.append({
                    "resume_name": resume_files[idx].filename,
                    "score": round(similarity_score, 2),
                    "email": emails[idx]
                })
            results = sorted(results, key=lambda x: x['score'], reverse=True)

            return render_template('index.html', results=results)

        # Delete the uploaded file after processing
    files = glob.glob("E:/Resume Screening/uploads/*")
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
