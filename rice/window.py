import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QWidget,
                             QMessageBox, QAction, QMenuBar, QGroupBox, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import torch.nn as nn
from torchvision import transforms

# 病虫害类别介绍
class_descriptions = {
    'Bacterial Leaf Blight': '水稻白叶枯病：由黄单胞杆菌引起，典型症状为叶片出现黄色至灰白色条斑，边缘有波浪状，严重时全叶枯死。',
    'Brown Spot': '水稻胡麻斑病：由平脐蠕孢菌引起，病斑呈椭圆形，褐色至深褐色，边缘明显，严重时导致叶片干枯。',
    'Healthy Rice Leaf': '健康水稻叶片：叶片颜色鲜绿，无病斑、虫蛀或其他异常症状，生长状态良好。',
    'Leaf Blast': '水稻叶瘟病：由稻瘟病菌引起，病斑呈梭形，中央灰白色，边缘褐色，潮湿时病斑上有灰绿色霉层。',
    'Leaf scald': '水稻叶鞘腐败病：由稻帚枝霉引起，主要危害叶鞘，初期病斑暗褐色，逐渐扩大呈云纹状，边缘褐色，中央淡褐色。',
    'Narrow Brown Leaf Spot': '水稻窄条斑病：由稻生尾孢菌引起，病斑窄长，褐色，多发生于叶片边缘，严重时可导致叶片早枯。',
    'Neck_Blast': '水稻穗颈瘟：由稻瘟病菌引起，危害穗颈、穗轴和枝梗，病部呈褐色或黑褐色，易折断，导致瘪粒甚至绝收。',
    'Rice Hispa': '水稻铁甲虫：成虫和幼虫均可危害，成虫啃食叶肉，留下白色条纹；幼虫潜入叶肉内取食，形成弯曲的虫道。',
    'Sheath Blight': '水稻纹枯病：由立枯丝核菌引起，主要危害叶鞘，病斑呈云纹状，边缘暗褐色，中央淡褐色，严重时可导致植株倒伏。'
}


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块，用于通道注意力"""

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """带SE注意力的残差块"""

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if self.use_se:
            self.se = SEModule(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super(ComplexCNN, self).__init__()

        # 初始卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 残差块组1
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)

        # 残差块组2
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # 残差块组3
        self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2)

        # 残差块组4
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

        # 空间金字塔池化
        self.spp = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = self.fc(x)
        return x


class CameraThread(QThread):
    """摄像头线程，用于实时捕获视频帧"""
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # 释放摄像头
        cap.release()

    def stop(self):
        """停止线程"""
        self._run_flag = False
        self.wait()


class RiceModelVisualizer(QMainWindow):
    """水稻模型可视化界面主窗口"""

    def __init__(self):
        super().__init__()

        # 设置窗口属性
        self.setWindowTitle("水稻病虫害识别系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                border: 1px solid #aaa;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                font-weight: bold;
            }
        """)

        # 初始化模型和摄像头
        self.model = None
        self.camera_thread = None
        self.is_camera_active = False

        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建标题
        title_label = QLabel("水稻病虫害识别系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("SimHei", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #333; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)

        # 创建功能区域
        self.create_function_area()

        # 创建结果显示区域
        self.create_result_area()

        # 状态栏
        self.statusBar().showMessage("就绪")

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        menubar.setStyleSheet("font-size: 14px;")

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 加载模型动作
        load_model_action = QAction('加载模型', self)
        load_model_action.triggered.connect(self.load_model)
        file_menu.addAction(load_model_action)

        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        # 使用说明动作
        usage_action = QAction('使用说明', self)
        usage_action.triggered.connect(self.show_usage)
        help_menu.addAction(usage_action)

        # 关于动作
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_function_area(self):
        """创建功能区域"""
        function_group = QGroupBox("操作区")
        function_layout = QHBoxLayout()
        function_group.setLayout(function_layout)

        # 按钮区域
        button_layout = QVBoxLayout()

        # 模型控制区
        model_group = QGroupBox("模型控制")
        model_layout = QHBoxLayout()
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setMinimumHeight(40)
        model_layout.addWidget(self.load_model_button)
        model_group.setLayout(model_layout)
        button_layout.addWidget(model_group)

        # 图像操作区
        image_group = QGroupBox("图像操作")
        image_layout = QVBoxLayout()

        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setEnabled(False)
        self.upload_button.setMinimumHeight(40)
        image_layout.addWidget(self.upload_button)

        self.camera_button = QPushButton("打开摄像头")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.camera_button.setEnabled(False)
        self.camera_button.setMinimumHeight(40)
        image_layout.addWidget(self.camera_button)

        self.capture_button = QPushButton("捕获图像")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        self.capture_button.setMinimumHeight(40)
        image_layout.addWidget(self.capture_button)

        image_group.setLayout(image_layout)
        button_layout.addWidget(image_group)

        button_layout.addStretch()
        function_layout.addLayout(button_layout, 1)

        # 图像显示区
        image_display_group = QGroupBox("图像显示")
        image_display_layout = QGridLayout()

        # 原始图像区域
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 400)
        self.original_label.setStyleSheet("""
            border: 2px solid #ddd; 
            border-radius: 5px; 
            background-color: #fff;
        """)
        image_display_layout.addWidget(self.original_label, 0, 0)

        # 结果图像区域
        self.result_label = QLabel("预测结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(500, 400)
        self.result_label.setStyleSheet("""
            border: 2px solid #ddd; 
            border-radius: 5px; 
            background-color: #fff;
        """)
        image_display_layout.addWidget(self.result_label, 0, 1)

        image_display_group.setLayout(image_display_layout)
        function_layout.addWidget(image_display_group, 3)

        self.main_layout.addWidget(function_group)

    def create_result_area(self):
        """创建结果显示区域"""
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout()

        # 结果文本显示
        self.result_text = QLabel("等待识别...")
        self.result_text.setAlignment(Qt.AlignCenter)
        self.result_text.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; margin-bottom: 10px;")
        result_layout.addWidget(self.result_text)

        # 病虫害介绍
        self.disease_description = QLabel("")
        self.disease_description.setWordWrap(True)
        self.disease_description.setStyleSheet("font-size: 14px; line-height: 1.5;")
        result_layout.addWidget(self.disease_description)

        # 置信度显示
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("font-size: 14px; margin-top: 10px;")
        result_layout.addWidget(self.confidence_label)

        result_group.setLayout(result_layout)
        self.main_layout.addWidget(result_group)

    def load_model(self):
        """加载PyTorch模型"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt)"
            )

            if file_path:
                # 加载完整模型（需确保模型定义可访问）
                self.model = torch.load(file_path, map_location=torch.device('cpu'))
                self.model.eval()  # 设置为评估模式
                self.statusBar().showMessage(f"已加载模型: {os.path.basename(file_path)}")
                self.upload_button.setEnabled(True)
                self.camera_button.setEnabled(True)

                # 显示模型信息
                QMessageBox.information(self, "成功", f"模型加载成功!\n\n模型文件: {os.path.basename(file_path)}")

                # 重置结果显示
                self.reset_result_display()

        except Exception as e:
            self.statusBar().showMessage("模型加载失败")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def upload_image(self):
        """上传图片并进行预测"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # 显示原始图像
            self.display_image(file_path, self.original_label)

            # 进行预测并显示结果
            self.predict_image(file_path)

    def toggle_camera(self):
        """打开或关闭摄像头"""
        if not self.is_camera_active:
            # 打开摄像头
            self.camera_thread = CameraThread()
            self.camera_thread.change_pixmap_signal.connect(self.update_camera_image)
            self.camera_thread.start()
            self.camera_button.setText("关闭摄像头")
            self.capture_button.setEnabled(True)
            self.is_camera_active = True
            self.statusBar().showMessage("摄像头已打开")
        else:
            # 关闭摄像头
            self.camera_thread.stop()
            self.camera_button.setText("打开摄像头")
            self.capture_button.setEnabled(False)
            self.is_camera_active = False
            self.original_label.setText("原始图像")
            self.statusBar().showMessage("摄像头已关闭")

    def update_camera_image(self, cv_img):
        """更新摄像头图像显示"""
        self.current_camera_frame = cv_img
        qt_img = self.convert_cv_to_qt(cv_img)
        self.original_label.setPixmap(qt_img.scaled(
            self.original_label.width(),
            self.original_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def convert_cv_to_qt(self, cv_img):
        """将OpenCV图像转换为Qt图像"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def capture_image(self):
        """捕获当前摄像头图像并进行预测"""
        if hasattr(self, 'current_camera_frame'):
            # 保存当前帧为临时文件
            temp_path = "temp_camera_capture.jpg"
            cv2.imwrite(temp_path, self.current_camera_frame)

            # 进行预测
            self.predict_image(temp_path)

    def display_image(self, image_path, label):
        """在指定标签中显示图像"""
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def predict_image(self, image_path):
        """使用模型预测图像并显示结果"""
        try:
            # 打开图像
            image = Image.open(image_path).convert('RGB')

            # 图像预处理（与训练流程对齐）
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image_tensor = transform(image).unsqueeze(0)  # 增加batch维度

            # 模型预测
            with torch.no_grad():
                output = self.model(image_tensor)

                # 获取预测类别和置信度
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                class_index = predicted.item()
                confidence_percent = confidence.item() * 100

                # 获取类别名称和描述
                class_names = [
                    'Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf',
                    'Leaf Blast', 'Leaf scald', 'Narrow Brown Leaf Spot',
                    'Neck_Blast', 'Rice Hispa', 'Sheath Blight'
                ]
                class_name = class_names[class_index]
                description = class_descriptions.get(class_name, "暂无该病虫害描述")

                # 结果文本
                result_text = f"预测结果: {class_name}"
                confidence_text = f"置信度: {confidence_percent:.2f}%"

                # 在图像上绘制预测结果
                result_image = cv2.imread(image_path)
                cv2.putText(
                    result_image, f"{result_text} ({confidence_text})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                # 保存并显示结果图像
                result_path = "temp_result.jpg"
                cv2.imwrite(result_path, result_image)
                self.display_image(result_path, self.result_label)

                # 更新结果显示
                self.result_text.setText(result_text)
                self.disease_description.setText(description)
                self.confidence_label.setText(confidence_text)

                self.statusBar().showMessage(f"预测完成: {result_text}")

        except Exception as e:
            self.statusBar().showMessage("预测失败")
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")

            # 重置结果显示
            self.reset_result_display()

    def reset_result_display(self):
        """重置结果显示"""
        self.result_text.setText("等待识别...")
        self.disease_description.setText("")
        self.confidence_label.setText("")

    def show_usage(self):
        """显示使用说明"""
        QMessageBox.information(self, "使用说明", """
水稻病虫害识别系统使用说明：

1. 首先点击"加载模型"按钮选择预训练模型文件(best_rice_model.pth)
2. 模型加载成功后，可以选择以下两种方式进行识别：
   - 点击"上传图片"按钮选择本地水稻叶片图片进行识别
   - 点击"打开摄像头"按钮使用摄像头进行实时检测，然后点击"捕获图像"进行识别
3. 识别结果将显示在下方区域，包括病虫害类型、详细描述和置信度
4. 支持识别的水稻病虫害包括：
   - 水稻白叶枯病
   - 水稻胡麻斑病
   - 水稻叶瘟病
   - 水稻叶鞘腐败病
   - 水稻窄条斑病
   - 水稻穗颈瘟
   - 水稻铁甲虫
   - 水稻纹枯病
""")

    def show_about(self):
        """显示关于信息"""
        QMessageBox.about(self, "关于", """
水稻病虫害识别系统 v1.0

该应用程序基于深度学习技术，用于识别水稻常见病虫害。
支持上传图片或使用摄像头进行实时检测，提供病虫害详细描述。

© 2025 开发者
""")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.is_camera_active:
            self.camera_thread.stop()
        event.accept()


if __name__ == "__main__":
    # 确保中文显示正常
    os.environ["QT_FONT_DPI"] = "96"

    app = QApplication(sys.argv)

    # 设置全局字体，确保中文正常显示
    font = QFont("SimHei")
    app.setFont(font)

    window = RiceModelVisualizer()
    window.show()

    sys.exit(app.exec_())