from flask import Flask, request, render_template
import os
import whisper

app = Flask(__name__)

UPLOADS_FOLDER = "./uploads"
model = whisper.load_model("base")

@app.route('/')
def index():
    file_names = get_file_names()
    return render_template('index.html', file_names=file_names)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件被上传
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # 检查文件名是否为空
        if file.filename == '':
            return 'No selected file'

        # 保存上传的文件
        file.save(f'uploads/{file.filename}')

    file_names = get_file_names()
    return render_template('index.html', file_names=file_names)

@app.route('/speech2text', methods=['POST'])
def speech2text():
    file_name = f'{UPLOADS_FOLDER}/{request.form["file_name"]}'

    result = model.transcribe(file_name)
    print(result["text"])

    return f'Transforming file: {file_name}'

def get_file_names():
    return os.listdir(UPLOADS_FOLDER)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

