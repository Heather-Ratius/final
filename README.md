# 吸烟行为检测系统

基于PaddlePaddle深度学习框架开发的吸烟行为检测系统，可以对图片和视频中的吸烟行为进行实时检测。

## 环境要求

- Python 3.7+
- PaddlePaddle 2.0+
- OpenCV
- Tkinter
- PIL
- numpy

## 安装依赖
pip install paddlepaddle-gpu
pip install opencv-python
pip install pillow
pip install numpy


3. 界面功能：
- 选择图片：支持jpg、jpeg、png格式
- 选择视频：支持mp4、avi、mov格式
- 停止检测：终止视频检测过程

## 模型说明

- 基础网络：ResNet50
- 检测头：自定义检测层
- 输出：目标边界框坐标 [xmin, ymin, xmax, ymax]

## 注意事项

1. 确保已安装所有依赖包
2. 运行detect_gui.py前需要有训练好的模型文件(smoke_detector_final.pdparams)
3. 视频检测支持实时暂停和继续
4. 检测结果会在图像/视频上绘制边界框

