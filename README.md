## Resume-Matching-Tool
### 📌 Overview

The Resume Matching Tool is an AI-powered web application that helps recruiters efficiently match resumes with job descriptions. Utilizing Natural Language Processing (NLP) techniques and machine learning algorithms, this tool ranks resumes based on their relevance to a given job posting.

### 🚀 Features

Automated resume parsing and text processing

Job description similarity scoring

Machine learning-based ranking using TF-IDF and K-Nearest Neighbors (KNN)

Cosine Similarity for document comparison

Web-based interface using Flask, HTML, CSS

Secure data storage with MySQL

### 🛠️ Tech Stack

Backend: Python, Flask

Frontend: HTML, CSS

Database: MySQL

Libraries & Tools:

NLP: NLTK

Machine Learning: TF-IDF, Sklearn-KNN

Similarity Measurement: Cosine Similarity


### ⚙️ Installation & Setup

1️⃣ Prerequisites

Ensure you have the following installed:

Python 3.x

MySQL

### 2️⃣ Clone the Repository

git clone https://github.com/SudipPatra35/Resume-Matching-Tool.git

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure the Database

Update config.py with your MySQL credentials.

Create the necessary tables using:

python models.py

5️⃣ Run the Application

python app.py

Access the tool at http://127.0.0.1:5000/

### 📊 How It Works

Upload a set of resumes (PDF/TXT format).

Provide a job description.

The system processes the text and ranks resumes based on similarity.

View ranked results with a matching percentage.

### 🔥 Future Enhancements

Support for more file formats (DOCX, JSON)

Advanced ML models (BERT, Word2Vec)

Integration with recruitment platforms

### 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.
