import os
import cv2
import whisper
from skimage import metrics
from moviepy.video.io.VideoFileClip import VideoFileClip
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)

class AutoMedia(object):
    def __init__(self, video_path, output_folder):
        """
        Parameters:
        - video_path: 视频输入文件 ./uploads/file_name.mp4
        - output_folder: 视频处理的输出位置 ./mids/
            - 进一步处理成 ./mids/file_name/
        """
        self.video_path = video_path
        self.output_folder = output_folder

        self.video_name = os.path.basename(video_path)
        self.frame_count = 0
        self.speech_file_names = []

        #self.model = whisper.load_model("medium")
        self.model = whisper.load_model("large")

    def get_frame_file_path(self, index):
        name = f'frame_{index}.jpg'
        return os.path.join(self.output_folder, name)

    def get_audio_file_path(self, st, et):
        name = f'audio_{st}_{et}.mp4'
        return os.path.join(self.output_folder, name)

    def get_speech_file_path(self, st, et):
        name = f'audio_{st}_{et}.txt'
        return os.path.join(self.output_folder, name)

    def extract_frames(self, video_path):
        """
        从视频中提取每秒的帧，并保存为图像文件。
        """
        frame_file_names = []

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logging.debug(f'video fps: {fps}')

        os.makedirs(self.output_folder, exist_ok=True)
        self.frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_count % fps == 0:
                frame_file_name = f'{self.frame_count // fps}'
                output_file = self.get_frame_file_path(frame_file_name)

                frame_file_names.append(output_file)
                cv2.imwrite(output_file, frame)
            self.frame_count += 1
        cap.release()
        return frame_file_names

    def find_breakpoint(self, image_path1, image_path2):
        """
        找断点
        重召回
        """
        if not os.path.exists(image_path1):
            return False
        if not os.path.exists(image_path2):
            return False

        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        # 确保两张图像具有相同的大小
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

        # 转换为灰度图像
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # 计算 SSIM
        ssim_score, _ = metrics.structural_similarity(image1_gray, image2_gray, full=True)
        logging.debug(f"{image_path1}-{image_path2} SSIM Score: {round(ssim_score, 2)}")
        if ssim_score < .2:
            return True
        return False

    def cal_video_vector(self, frame_file_names):
        breakpoints = []
        for i in range(len(frame_file_names) - 1):
            frame_path1 = frame_file_names[i]
            frame_path2 = frame_file_names[i + 1]
            is_breakpoint = self.find_breakpoint(frame_path1, frame_path2)
            if is_breakpoint:
                breakpoints.append(i)
        return breakpoints

    def speech2text(self, input_file_name, output_file_name):
        prompt='以下是普通话的句子'
        result = self.model.transcribe(input_file_name, language='zh', initial_prompt=prompt)
        result_text = result['text']
        with open(output_file_name, 'w', encoding='utf-8') as out:
            out.write(result_text)

    def process_video(self, video_path, output_path):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 获取视频的帧宽度和帧高度
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # 计算要去除的底部高度
        trim_top = int(frame_height * 0.05)
        trim_buttom = int(frame_height * 0.1)
        trim_left = int(frame_width * 0.05)
        trim_right = int(frame_width * 0.05)

        # 设置视频编解码器和输出视频对象
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width - trim_left - trim_right, frame_height - trim_top - trim_buttom))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 去除底部5%的图像内容
            trimmed_frame = frame[trim_top: -trim_buttom, trim_left: -trim_right]

            # 写入输出视频
            out.write(trimmed_frame)

            cv2.imshow('Trimmed Video', trimmed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def extract_video_segment(self, video_path, start_time, end_time, output_path):
        # 打开视频文件
        video_clip = VideoFileClip(video_path)

        # 切分指定时间范围的视频片段
        cut_clip = video_clip.subclip(start_time, end_time)

        # 保存视频片段
        cut_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # 关闭视频文件
        video_clip.close()

    def cal_audio_vector(self, cost, input_path, output_path):
        lines = []
        total_words = 0
        with open(input_path) as fin:
            for line in fin:
                line = line.strip()
                lines.append(line)
                total_words += len(line)

        acc_cost = 0
        prts = []
        for line in lines:
            length = len(line)
            ratio = 1. * length / total_words
            line_cost = cost * ratio
            acc_cost += line_cost
            prt = '%s\t%s\n' % (line, acc_cost)
            prts.append(prt)

        with open(output_path, 'w') as fout:
            for line in prts:
                fout.write(line)

    def write_video_vector(self, vec):
        file_path = 'info_video.txt'
        msg = ','.join(map(str, vec))
        with open(file_path, 'w') as fout:
            fout.write(msg)
            fout.write('\n')

def process_audio(file_path, mid_path):
    mid_path = './mids'
    #process_step1(file_path, mid_path)
    process_step2(file_path, mid_path)

def process_step1(file_path, mid_path):
    auto = AutoMedia(file_path, mid_path)
    auto.speech2text(file_path, 'info_audio1.txt')
    # 放入ChatGPT等工具中，进行断句，生成txt_step2.txt

def process_step2(file_path, mid_path):
    """
    Params:
        total time cost
        text file
    """
    cost = 322
    auto = AutoMedia(file_path, mid_path)
    auto.cal_audio_vector(cost, 'info_audio2.txt', 'info_audio3.txt')

def process_video(file_path, mid_path):
    auto = AutoMedia(file_path, mid_path)
    frame_file_names = auto.extract_frames(file_path)
    video_vector = auto.cal_video_vector(frame_file_names)
    auto.write_video_vector(video_vector)
    print(video_vector)

def process_match(audio_path, video_path):
    audio_map = {}
    with open(audio_path) as fin:
        for line in fin:
            line = line.strip('\n').split('\t')
            if len(line) < 2:
                continue
            content = line[0]
            sec = int(float(line[1]))
            audio_map[sec] = content
    print(audio_map)

    # 30min hashmap
    video_hashmap = [0] * 1800
    with open(video_path) as fin:
        for line in fin:
            line = line.strip('\n').split(',')
            for k in line:
                video_hashmap[int(k)] = 1
    print(video_hashmap)

    match_vector = []
    for k, v in audio_map.items():
        offsets = [0, 1, 2, -1, -2]
        for i in offsets:
            sec = k + i
            if video_hashmap[sec]:
                match_vector.append(k)
                break
    return match_vector

def process_cluster(arr, num, st, et):
    arr.append(st)
    arr.append(et)
    arr.sort()
    while len(arr) > num:
        min_gap = float('inf')
        merge_index = 0
        for i in range(len(arr) - 2):
            j = i + 2
            gap = arr[j] - arr[i]
            if gap < min_gap:
                min_gap = gap
                merge_index = i + 1
        arr.pop(merge_index)
    return arr

def process_media(ori_path, audio_path, vector, mid_path):
    # audio_map存储的是截止到ns前的文本内容
    # vector存储的是时间切片
    audio_map = [''] * 1800
    with open(audio_path) as fin:
        for line in fin:
            line = line.strip('\n').split('\t')
            if len(line) < 2:
                continue
            content = line[0]
            sec = int(float(line[1]))
            audio_map[sec] = content
    print(audio_map)
    print(vector)

    for i in range(len(vector) - 1):
        start_time = vector[i] + 1
        end_time = vector[i + 1] + 1
        content = ''.join(audio_map[start_time : end_time])
        duration = end_time - start_time
        output_file = '%s/video_seg_%s.mp4' % (mid_path, i)
        command = f'ffmpeg -ss {start_time} -t {duration} -i {ori_path} -c:v libx264 -c:a aac -strict experimental -b:a 192k {output_file}'
        subprocess.run(command, shell=True)
        output_file = '%s/content_seg_%s.txt' % (mid_path, i)
        with open(output_file, 'w') as fout:
            fout.write(content)
            fout.write('\n')

if __name__ == "__main__":
    file_path = 'uploads/锈铁1.mp4'
    mid_path = './mids'
    #process_audio(file_path, mid_path)
    #process_video(file_path, mid_path)
    match_vector = process_match('info_audio3.txt', 'info_video.txt')
    print(match_vector)
    cluster_vector = process_cluster(match_vector, 10, 0, 322)
    print(cluster_vector)
    process_media(file_path, 'info_audio3.txt', cluster_vector, mid_path)



