import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import paddle
import paddle.vision.transforms as T
from PIL import Image, ImageTk
import os
from paddle.vision.models import resnet50

class SmokeDetector(paddle.nn.Layer):
    def __init__(self):
        super(SmokeDetector, self).__init__()
        
        backbone = resnet50(pretrained=True)
        self.features = paddle.nn.Sequential(*list(backbone.children())[:-2])
        
        self.detector = paddle.nn.Sequential(
            paddle.nn.Conv2D(2048, 512, 1),
            paddle.nn.ReLU(),
            paddle.nn.AdaptiveAvgPool2D(1),
            paddle.nn.Flatten(),
            paddle.nn.Linear(512, 4)
        )
        
    def forward(self, x):
        features = self.features(x)
        bbox = self.detector(features)
        return bbox

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("吸烟行为检测系统")
        self.root.geometry("1200x800")
        
        self.model = SmokeDetector()
        self.model.eval()
        # 加载训练好的模型权重
        self.model.set_state_dict(paddle.load('smoke_detector_final.pdparams'))
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_gui()
        
    def setup_gui(self):
        # 主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 按钮
        ttk.Button(self.main_frame, text="选择图片", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.main_frame, text="选择视频", command=self.load_video).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.main_frame, text="停止检测", command=self.stop_detection).grid(row=0, column=2, padx=5, pady=5)
        
        # 显示区域
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600)
        self.canvas.grid(row=1, column=0, columnspan=3, pady=10)
        
        # 状态标签
        self.status_label = ttk.Label(self.main_frame, text="就绪")
        self.status_label.grid(row=2, column=0, columnspan=3)
        
        self.is_detecting = False
        self.current_video = None
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.detect_image(file_path)
    
    def detect_image(self, image_path):
        # 读取图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # 预处理图片
        img_pil = Image.fromarray(image)
        img_tensor = self.transform(img_pil)
        img_tensor = paddle.unsqueeze(img_tensor, 0)
        
        # 检测
        with paddle.no_grad():
            bbox = self.model(img_tensor)
            bbox = bbox.numpy()[0]
        
        # 转换坐标到原始图片大小
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * original_w)
        ymin = int(ymin * original_h)
        xmax = int(xmax * original_w)
        ymax = int(ymax * original_h)
        
        # 绘制边界框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, 'Smoking', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 显示结果
        self.show_image(image)
        
    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.MOV")]
        )
        if file_path:
            self.is_detecting = True
            self.current_video = cv2.VideoCapture(file_path)

            if not self.current_video.isOpened():
                self.status_label.config(text="无法打开视频文件")
                return
            
            self.detect_video()
    
    def detect_video(self):
        if not self.is_detecting:
            self.current_video.release()
            return
        
        ret, frame = self.current_video.read()
        if ret:
            # 处理帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_h, original_w = frame_rgb.shape[:2]
            
            # 预处理
            img_pil = Image.fromarray(frame_rgb)
            img_tensor = self.transform(img_pil)
            img_tensor = paddle.unsqueeze(img_tensor, 0)
            
            # 检测
            with paddle.no_grad():
                bbox = self.model(img_tensor)
                bbox = bbox.numpy()[0]
            
            # 转换坐标
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * original_w)
            ymin = int(ymin * original_h)
            xmax = int(xmax * original_w)
            ymax = int(ymax * original_h)
            
            # 边界框
            cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame_rgb, 'Smoking', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 显示结果
            self.show_image(frame_rgb)
            
            # 继续处理下一帧
            self.root.after(30, self.detect_video)
        else:
            self.current_video.release()
            self.status_label.config(text="视频处理完成")
    
    def show_image(self, image):
        h, w = image.shape[:2]
        canvas_w = 800
        canvas_h = 600
        
        # 保持宽高比
        ratio = min(canvas_w/w, canvas_h/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        image = cv2.resize(image, (new_w, new_h))

        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo 
    
    def stop_detection(self):
        self.is_detecting = False
        if self.current_video is not None:
            self.current_video.release()
        self.status_label.config(text="检测已停止")

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop() 