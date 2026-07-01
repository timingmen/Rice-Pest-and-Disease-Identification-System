import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch.nn as nn
from torchvision import transforms

# 确保中文显示正常
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

app = Flask(__name__)
CORS(app)  # 启用CORS支持跨域请求

# 设置上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

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

# 防治建议
treatment_advice = {
    'Bacterial Leaf Blight': '1. 选用抗病品种；2. 种子消毒处理；3. 发病初期喷施叶枯唑、噻菌铜等杀菌剂；4. 及时清除病叶和病株，减少病原菌传播。',
    'Brown Spot': '1. 种子消毒；2. 合理密植，保持田间通风透光；3. 增施有机肥和磷钾肥，提高植株抗病性；4. 发病初期喷施多菌灵、三环唑等杀菌剂。',
    'Healthy Rice Leaf': '1. 保持田间合理灌溉和排水；2. 定期监测稻田，预防病虫害发生；3. 合理施肥，增强水稻抵抗力。',
    'Leaf Blast': '1. 选用抗病品种；2. 种子消毒；3. 合理密植，科学施肥；4. 发病初期喷施三环唑、稻瘟灵等杀菌剂。',
    'Leaf scald': '1. 清除病残体，减少病原菌越冬；2. 合理密植，改善通风透光条件；3. 发病初期喷施多菌灵、甲基硫菌灵等杀菌剂。',
    'Narrow Brown Leaf Spot': '1. 加强田间管理，合理排灌；2. 及时清除病叶；3. 发病初期喷施噻森铜、噻菌铜等杀菌剂。',
    'Neck_Blast': '1. 选用抗病品种；2. 种子消毒；3. 破口期和齐穗期各喷施一次三环唑、稻瘟灵等杀菌剂；4. 及时清除病穗。',
    'Rice Hispa': '1. 释放赤眼蜂等天敌；2. 使用苏云金芽孢杆菌（Bt）等生物农药；3. 化学防治可选用氯虫苯甲酰胺、甲氨基阿维菌素苯甲酸盐等杀虫剂。',
    'Sheath Blight': '1. 合理密植，改善通风透光条件；2. 控制氮肥用量，增施磷钾肥；3. 发病初期喷施井冈霉素、噻呋酰胺等杀菌剂；4. 及时清除病叶和菌核。'
}

# 病害严重程度映射
severity_map = {
    'Bacterial Leaf Blight': '高',
    'Brown Spot': '中',
    'Healthy Rice Leaf': '无',
    'Leaf Blast': '高',
    'Leaf scald': '中',
    'Narrow Brown Leaf Spot': '低',
    'Neck_Blast': '高',
    'Rice Hispa': '中',
    'Sheath Blight': '中'
}

# 严重程度颜色映射
severity_color_map = {
    '高': 'text-danger',
    '中': 'text-warning',
    '低': 'text-neutral',
    '无': 'text-secondary'
}


# 模型定义
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


# 初始化模型
model = None
class_names = [
    'Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf',
    'Leaf Blast', 'Leaf scald', 'Narrow Brown Leaf Spot',
    'Neck_Blast', 'Rice Hispa', 'Sheath Blight'
]


# 主页路由
@app.route('/')
def index():
    return render_template('index.html')


# 加载模型API
@app.route('/api/load_model', methods=['POST'])
def load_model():
    global model
    try:
        model_path = request.json.get('model_path')
        if not model_path:
            return jsonify({'status': 'error', 'message': '未指定模型路径'})

        # 加载模型
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        return jsonify({'status': 'success', 'message': f'模型加载成功: {os.path.basename(model_path)}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'模型加载失败: {str(e)}'})


# 上传图片并预测API
@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': '模型未加载'})

    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': '未上传图片'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择图片'})

        # 保存上传的图片
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # 打开图像
        image = Image.open(image_path).convert('RGB')

        # 图像预处理
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
            output = model(image_tensor)

            # 获取预测类别和置信度
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            class_index = predicted.item()
            confidence_percent = confidence.item() * 100

            # 获取类别名称和描述
            class_name = class_names[class_index]
            description = class_descriptions.get(class_name, "暂无该病虫害描述")
            treatment = treatment_advice.get(class_name, "暂无防治建议")
            severity = severity_map.get(class_name, "中")
            severity_color = severity_color_map.get(severity, "text-neutral")

            # 在图像上绘制预测结果
            result_image = cv2.imread(image_path)
            cv2.putText(
                result_image, f"{class_name} ({confidence_percent:.2f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 保存结果图像
            result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
            cv2.imwrite(result_path, result_image)

            # 返回结果
            return jsonify({
                'status': 'success',
                'disease_name': class_name,
                'description': description,
                'treatment': treatment,
                'confidence': f"{confidence_percent:.2f}%",
                'severity': severity,
                'severity_color': severity_color,
                'original_image_url': f'/uploads/{file.filename}',
                'result_image_url': f'/results/{file.filename}'
            })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'预测失败: {str(e)}'})


# 上传文件夹路由
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# 结果文件夹路由
@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)