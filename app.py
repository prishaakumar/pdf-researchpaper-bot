from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from rag_pipeline import process_pdf_bytes_and_answer

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = error = None
    if request.method == 'POST':
        if 'pdf' not in request.files or 'question' not in request.form:
            error = 'PDF file and question required.'
        elif not request.files['pdf'].filename:
            error = 'Please select a PDF file.'
        else:
            pdf_file = request.files['pdf']
            question = request.form['question']
            try:
                pdf_bytes = pdf_file.read()
                if not pdf_bytes:
                    error = 'Uploaded file is empty.'
                else:
                    answer = process_pdf_bytes_and_answer(pdf_bytes, question)
            except Exception as e:
                error = f'Application error: {str(e)}'
    return render_template('index.html', answer=answer, error=error)


@app.route('/ask', methods=['POST'])
def ask():
    if 'pdf' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'PDF file and question required'}), 400
    pdf_file = request.files['pdf']
    if not pdf_file.filename:
        return jsonify({'error': 'Please select a PDF file.'}), 400
    question = request.form['question']
    try:
        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            return jsonify({'error': 'Uploaded file is empty.'}), 400
        answer = process_pdf_bytes_and_answer(pdf_bytes, question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Force local port 15000 so it matches your browser URL
    app.run(host='127.0.0.1', port=15000, debug=False)