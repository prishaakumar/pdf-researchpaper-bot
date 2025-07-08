from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template
import os
from rag_pipeline import process_pdf_and_answer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = error = None
    if request.method == 'POST':
        if 'pdf' not in request.files or 'question' not in request.form:
            error = 'PDF file and question required.'
        else:
            pdf_file = request.files['pdf']
            question = request.form['question']
            pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
            pdf_file.save(pdf_path)
            try:
                answer = process_pdf_and_answer(pdf_path, question)
            except Exception as e:
                error = str(e)
    return render_template('index.html', answer=answer, error=error)

@app.route('/ask', methods=['POST'])
def ask():
    if 'pdf' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'PDF file and question required'}), 400
    pdf_file = request.files['pdf']
    question = request.form['question']
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)
    try:
        answer = process_pdf_and_answer(pdf_path, question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 