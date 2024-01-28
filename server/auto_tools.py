import os
import cv2
from skimage import metrics

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

    def get_frame_file_path(self, index):
        name = f'frame_{index}.jpg'
        return os.path.join(self.output_folder, name)

    def extract_frames(self, video_path):
        """
        从视频中提取每秒的帧，并保存为图像文件。
        """
        frame_file_names = []

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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
        if ssim_score < .1:
            print(f"{image_path1}-{image_path2} SSIM Score: {round(ssim_score, 2)}")
            return True
        return False

    def cal_breakpoints(self, frame_file_names):
        breakpoints = []
        for i in range(len(frame_file_names) - 1):
            frame_path1 = frame_file_names[i]
            frame_path2 = frame_file_names[i + 1]
            is_breakpoint = self.find_breakpoint(frame_path1, frame_path2)
            if is_breakpoint:
                breakpoints.append(i)
        return breakpoints

    def cluster_breakpoints(self, breakpoints):
        """
        聚合10s以内的断点
        """
        clustered_pts = []
        clustered_pts.append(breakpoints[0])

        last_pt = breakpoints[0]
        for i in range(1, len(breakpoints)):
            pt = breakpoints[i]
            if abs(pt - last_pt) > 10:
                clustered_pts.append(pt)
                last_pt = pt
        print(breakpoints)
        print(clustered_pts)
        return clustered_pts

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

    def process_sounds(self, vodeo_path, output_path):
        frame_file_names = self.extract_frames(self.video_path)
        breakpoints = self.cal_breakpoints(frame_file_names)
        breakpoints = self.cluster_breakpoints(breakpoints)
        print(breakpoints)

if __name__ == "__main__":
    upload_path = './uploads'
    mid_path = './mids'

    for file_name in os.listdir(upload_path):
        file_path = os.path.join(upload_path, file_name)

        # 检查是否为MP4文件
        if file_name.endswith(".mp4"):
            # file_name: video.mp4
            print(f"正在处理文件: {file_path}")
            auto = AutoMedia(file_path, mid_path)
            auto.process_video(file_path, 'a.mp4')
            #auto.process_sounds(file_path, 'a.txt')


