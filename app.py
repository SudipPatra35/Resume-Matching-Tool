from flask import Flask, get_flashed_messages, render_template, request, redirect, url_for, flash, session, send_from_directory
import os
import re
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
import mysql.connector
import pymysql


# Initialize Flask app and session management
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.secret_key = 'secret1234'
app.config['SESSION_TYPE'] = 'filesystem'
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Admin Credentials (hardcoded, without encryption)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin12345"
emptyMessage = 1


# MySQL Database Connection
def get_db_connection():
    connection = pymysql.connect(
    host="localhost",
    user="root",
    password="Sudip@7797",
    database="resumetool",
    port=3306
    )
    return connection


@app.route('/')
def start():
    return render_template('login.html')


# Route for Login (User and Admin)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        universityRoll = request.form['universityRoll']
        password = request.form['password']

        # Admin login check (statically check against hardcoded admin credentials)
        if universityRoll == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_id'] = 1  # Set admin session ID
            # flash('Logged in successfully as admin', 'success')
            return redirect(url_for('admin_page'))

        # User login check (check from the database)
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE universityRoll = %s", (universityRoll,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()

        if user and user[7] == password:  # Assuming user[2] is the password field
            session['universityRoll'] = user[0]  # Store user email in session
            # flash('User logged in successfully!', 'success')
            return redirect(url_for('student', user=user))

        flash('Invalid University Roll or Password', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        universityRoll = request.form['universityRoll']
        name = request.form['name']
        gender = request.form['gender']
        mobile = request.form['mobile']
        email = request.form['email']
        degree = request.form['degree']
        department = request.form['department']
        password = request.form['password']

        # Hash the password for secure storage

        # Establish database connection
        connection = get_db_connection()
        cursor = connection.cursor()

        try:
            # Check if the universityRoll already exists
            cursor.execute("SELECT * FROM users WHERE universityRoll = %s", (universityRoll,))
            existing_user = cursor.fetchone()

            if existing_user:
                print("User already exists.")
                flash('University Roll already exists! Please log in.', 'danger')
                return redirect(url_for('login'))

            # Insert the new user into the database
            cursor.execute("""
                INSERT INTO users (universityRoll, password, name, gender, mobile, email, degree, department)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (universityRoll,password, name, gender, mobile, email, degree, department))
            connection.commit()

            flash('Registration successful! Welcome, please log in.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            # Log the error for debugging (optional)
            print(f"Error during registration: {e}")
            flash('An unexpected error occurred. Please try again later.', 'danger')

        finally:
            cursor.close()
            connection.close()

    return render_template('register.html')


# Route for User page to upload resumes
@app.route('/student', methods=['GET', 'POST'])
def student():
    global emptyMessage  # Ensure you're using this flag to control state
    # Ensure the user is logged in
    if 'universityRoll' not in session:
        flash('You need to login first', 'danger')
        return redirect(url_for('start'))

    # Retrieve user details from the database            
    universityRoll = session['universityRoll']
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE universityRoll = %s", (universityRoll,))
    resume = cursor.fetchone()

    # Clear any unnecessary old flash messages
    if resume:
        if resume[9] == "PENDING":
            flash("Resume uploaded successfully! Awaiting approval. Please wait for approval.", "info")
        elif resume[9] == "FALSE":
            flash("Your resume was rejected. Please re-upload your resume.", "danger")
        elif resume[9] == "empty" and emptyMessage == 1:
            flash("No resume found. Please upload your resume.", "warning")
            # After flashing the "empty" message, reset the flag so it doesn't show again
            emptyMessage = 0  # Prevent this message from showing again
        elif resume[9] == "TRUE":
            flash("Your resume has been approved.", "success")

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'resume' not in request.files:
            flash('No file part provided in the request.', 'danger')
            return redirect(request.url)

        file = request.files['resume']

        # Validate the uploaded file
        if file.filename == '':
            flash('No file selected. Please choose a file to upload.', 'danger')
            return redirect(request.url)

        if file:
            # Save the file to the uploads folder
            file_extension = file.filename.split('.')[-1]
            filename = f"{universityRoll}.{file_extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Update the user's record in the database
            cursor.execute("UPDATE users SET filename = %s, approved = 'PENDING' WHERE universityRoll = %s",
                           (filename, universityRoll))
            # flash("Resume uploaded successfully! Awaiting approval. Please wait for approval.", "success")
            connection.commit()

            # After a successful upload, reset emptyMessage to prevent the "empty" flash
            emptyMessage = 0

            return redirect(url_for('student'))

    # Close the database connection
    cursor.close()
    connection.close()

    return render_template('student.html', user=resume)


@app.route('/admin', methods=['GET'])
def admin_page():
    connection = get_db_connection()
    cursor = connection.cursor()

    # Fetch all unapproved resumes (adjust your query as needed)
    cursor.execute("SELECT universityRoll,name FROM users WHERE approved = 'PENDING'")
    resumes = cursor.fetchall()
    # print(resumes)

    cursor.close()
    connection.close()

    # Pass the fetched resumes to the template
    return render_template('admin.html',resumes=resumes)


@app.route('/uploads/<user_id>')
def uploaded_file(user_id):
    # Check for existing files with supported extensions
    for ext in ['pdf', 'docx', 'txt']:  # Add other file extensions as needed
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}.{ext}")
        if os.path.exists(file_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], f"{user_id}.{ext}")

    return "File not found", 404


# Route to approve or reject a resume
@app.route('/approve', methods=['POST'])
def approve_resumes():
    if 'admin_id' not in session:
        flash('You must log in as an admin to approve resumes', 'danger')
        return redirect(url_for('index'))

    # Fetch selected resumes from the form
    selected_resumes = request.form.getlist('selected_resumes')
    if not selected_resumes:
        flash('No resumes selected for approval', 'danger')
        return redirect(url_for('admin_page'))

    connection = get_db_connection()
    cursor = connection.cursor()

    # Update the database to set selected resumes as approved
    try:
        for resume_id in selected_resumes:
            cursor.execute("UPDATE users SET approved = 'TRUE' WHERE universityRoll = %s", (resume_id,))
        connection.commit()
        flash('Selected resumes have been approved!', 'success')
    except Exception as e:
        connection.rollback()
        flash(f'An error occurred: {str(e)}', 'danger')
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for('admin_page'))



@app.route('/reject', methods=['POST'])
def reject_resumes():
    if 'admin_id' not in session:
        flash('You must log in as an admin to reject resumes', 'danger')
        return redirect(url_for('index'))

    selected_resumes = request.form.getlist('selected_resumes')

    if not selected_resumes:
        flash('No resumes selected for rejection', 'danger')
        return redirect(url_for('admin_page'))

    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT filename FROM users WHERE universityRoll IN %s", (list(selected_resumes),))
    resumes_to_delete = cursor.fetchall()

    # Update the resumes as rejected
    cursor.execute("UPDATE users SET approved = 'FALSE' WHERE universityRoll IN %s", (tuple(selected_resumes),))
    connection.commit()

    for resume in resumes_to_delete:
        # print("resumes_to_delete",resume[0])
        # filename = resume # Fetch filename from the query result
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],resume[0])

        # Check if the file exists before attempting to delete
        if os.path.exists(file_path):
            os.remove(file_path)

    cursor.close()
    connection.close()

    flash('Selected resumes have been rejected!', 'danger')
    return redirect(url_for('admin_page'))


# Route for Resume Matching
@app.route('/matching', methods=['GET'])
def matching():
    # When the page is loaded, fetch approved resumes and render the matching page
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE approved = 'TRUE'")
    approved_resumes = cursor.fetchall()
    # print(approved_resumes)
    # Store the approved resumes in session
    session['approved_resumes'] = approved_resumes

    # Render the matching page with the list of approved resumes
    return render_template('matching.html', approved_resumes=approved_resumes)


@app.route('/match_resumes', methods=['POST'])
def match_resumes():
    job_description = request.form['jobDescription']
    top_resume_no = int(request.form["resume_no"])

    # Retrieve the approved resumes from the session
    approved_resumes = session.get('approved_resumes', [])

    if len(approved_resumes) < top_resume_no:
        flash("Please approve more resumes for matching.", "warning")
        return redirect(url_for('matching'))  # Redirect back to the matching page if not enough resumes

    # Extract resume text and emails for approved resumes
    resumes = []
    emails = []
    for resume in approved_resumes:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], resume[8])
        # You would need to implement extract_text() to extract the content from the resume
        email, resume_text = extract_text(filename)
        resumes.append(resume_text)
        emails.append(email)

    # Preprocess the job description (ensure it's cleaned and ready)
    job_description = data_preprocessing(job_description)

    # Convert job description and resumes to vectors using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
    vectors = vectorizer.toarray()

    # Extract vectors for job description and resumes
    job_vector = vectors[0]
    resume_vector = vectors[1:]

    # KNN model to find closest resumes based on cosine similarity
    knn = NearestNeighbors(n_neighbors=top_resume_no, metric='cosine')
    knn.fit(resume_vector)  # Fit the KNN model on resume vectors only
    distances, indices = knn.kneighbors([job_vector])

    # Prepare results with resume names, scores, and emails
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        similarity_score = ((1 - distances[0][i]) * 100)  # Convert distance to similarity percentage
        similarity_score = (similarity_score * 100) // 22  # Adjust the scaling if necessary
        if similarity_score >=99 : 
            similarity_score = 95 
        results.append({
            "student_name": approved_resumes[idx][1],
            "score": round(similarity_score, 2),
            # "email": emails[idx]
            "userID" :  approved_resumes[idx][0]
        })

    # Sort results by similarity score in descending order
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Pass results to the template
    return render_template('matching.html', results=results)


nltk_data_path = "C:/Users/Sudip/Desktop/My Project/nltk_data"
nltk.data.path.append(nltk_data_path)

#run below 3 line only first time...
# nltk.download('punkt_tab',download_dir=nltk_data_path)
# nltk.download('wordnet',download_dir=nltk_data_path)
# nltk.download('stopwords',download_dir=nltk_data_path)
# nltk.download('averaged_perceptron_tagger_eng',download_dir=nltk_data_path)


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
    tokens=[]
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


if __name__ == '__main__':
    app.run(debug=True)
