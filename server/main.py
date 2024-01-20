from flask import Flask, request, render_template
import os
import whisper
import time

app = Flask(__name__)

UPLOADS_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
"""
Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
tiny	39 M	tiny.en	tiny	~1 GB	~32x
base	74 M	base.en	base	~1 GB	~16x
small	244 M	small.en	small	~2 GB	~6x
medium	769 M	medium.en	medium	~5 GB	~2x
large	1550 M	N/A	large	~10 GB	1x
"""
model = whisper.load_model("small")

def get_upload_file_names():
    return os.listdir(UPLOADS_FOLDER)

def get_all_file_names():
    resps = []
    upload_file_names = get_upload_file_names()
    for upload_file_name in upload_file_names:
        name = upload_file_name
        try:
            name = os.path.splitext(upload_file_name)[0]
        except:
            pass

        mp4_name = ""
        txt_name = ""
        variant_mp4_name = ""
        variant_txt_name = ""
        check_file_name = f'{UPLOADS_FOLDER}/{name}.mp4'
        if os.path.exists(check_file_name):
            mp4_name = upload_file_name

        check_file_name = f'{OUTPUT_FOLDER}/{name}.txt'
        if os.path.exists(check_file_name):
            txt_name = 'Done'

        check_file_name = f'{OUTPUT_FOLDER}/{name}_variant.txt'
        if os.path.exists(check_file_name):
            variant_txt_name = 'Done'

        check_file_name = f'{OUTPUT_FOLDER}/{name}_variant.mp4'
        if os.path.exists(check_file_name):
            variant_mp4_name = 'Done'
        resp = {}
        resp['mp4_name'] = mp4_name
        resp['txt_name'] = txt_name
        resp['variant_txt_name'] = variant_txt_name
        resp['variant_mp4_name'] = variant_mp4_name
        resps.append(resp)
    resps = {index: resp for index, resp in enumerate(resps)}
    return resps

@app.route('/')
def index():
    file_names = get_all_file_names()
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

    file_names = get_all_file_names()
    return render_template('index.html', file_names=file_names)

@app.route('/speech2text', methods=['POST'])
def speech2text():
    print(request.form)
    file_name = request.form["file_name"]

    name = file_name
    try:
        name = os.path.splitext(name)[0]
    except:
        pass

    st = time.time()
    upload_file_name = f'{UPLOADS_FOLDER}/{file_name}'
    text_file_name = f'{OUTPUT_FOLDER}/{name}.txt'

    prompt='以下是普通话的句子'
    result = model.transcribe(upload_file_name, language='zh', initial_prompt=prompt)
    result_text = result['text']

    with open(text_file_name, 'w', encoding='utf-8') as out:
        out.write(result_text)

    et = time.time()
    cost = int(et - st)
    print(result_text, cost)

    file_names = get_all_file_names()
    return render_template('index.html', file_names=file_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

