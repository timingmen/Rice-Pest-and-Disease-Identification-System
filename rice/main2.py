import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageEnhance
import shutil
from collections import defaultdict
import onnx
#import onnxruntime
import tensorflow as tf
from onnx_tf.backend import prepare

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优


def get_gpu_memory():
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        stats = torch.cuda.memory_stats()
        memory_allocated = stats['allocated_bytes.all.current'] / 1024 ** 2
        memory_cached = stats['reserved_bytes.all.current'] / 1024 ** 2
        return f"GPU内存使用: {memory_allocated:.2f} MB (缓存: {memory_cached:.2f} MB)"
    return "无可用GPU"


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


# 自定义安全的色调调整函数
def safe_adjust_hue(img, hue_factor):
    """安全调整PIL图像的色调，避免溢出错误"""
    # 转换为HSV色彩空间
    img = img.convert('HSV')
    h, s, v = img.split()

    # 调整色调，使用浮点数计算避免溢出
    np_h = np.array(h, dtype=np.int32)
    np_h = (np_h + int(hue_factor * 255)) % 256  # 确保结果在0-255范围内
    h = Image.fromarray(np_h.astype(np.uint8))

    # 合并回HSV并转换回RGB
    img = Image.merge('HSV', (h, s, v))
    return img.convert('RGB')


class SafeHueTransform:
    """自定义变换类，安全地调整图像色调"""

    def __init__(self, hue_factor_range=(-0.05, 0.05)):
        self.hue_factor_range = hue_factor_range

    def __call__(self, img):
        # 随机生成hue_factor
        hue_factor = np.random.uniform(*self.hue_factor_range)
        return safe_adjust_hue(img, hue_factor)


class RiceDiseaseClassifier:
    def __init__(self, data_dir, model_save_path="best_model.pth", batch_size=32, num_workers=4, gpu_id=0):
        """
        初始化水稻病害分类器
        :param data_dir: 数据集目录，应包含所有类别的图像
        :param model_save_path: 模型保存路径
        :param batch_size: 批次大小
        :param num_workers: 数据加载的工作进程数
        :param gpu_id: 指定使用的GPU ID
        """
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.gpu_id = gpu_id
        self.classes = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # 打印GPU信息
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU内存总量: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 2:.2f} MB")
        else:
            print("警告: 未检测到GPU，将使用CPU运行。训练可能会非常缓慢。")

    def prepare_data(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        """
        准备数据集，自动将数据划分为训练集、验证集和测试集
        :param train_ratio: 训练集比例
        :param val_ratio: 验证集比例
        :param test_ratio: 测试集比例
        :param seed: 随机种子
        """
        # 确保比例之和为1
        # assert train_ratio + val_ratio + test_ratio == 1.0

        # 使用自定义安全的色调调整
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            SafeHueTransform(hue_factor_range=(-0.05, 0.05)),  # 使用自定义安全色调调整
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 验证集和测试集使用相同的变换，不进行数据增强
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载完整数据集
        full_dataset = datasets.ImageFolder(self.data_dir)
        self.classes = full_dataset.classes
        print(f"发现{len(self.classes)}个类别: {self.classes}")

        # 按类别统计样本数量
        class_counts = defaultdict(int)
        for _, label in full_dataset:
            class_counts[label] += 1

        # 打印每个类别的样本数量
        print("\n各类别样本数量:")
        for idx, class_name in enumerate(self.classes):
            print(f"  {class_name}: {class_counts[idx]}张图像")

        # 计算每个类别的样本数量
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        print(f"\n数据集划分比例: 训练集 {train_ratio:.0%}, 验证集 {val_ratio:.0%}, 测试集 {test_ratio:.0%}")
        print(f"数据集大小: 总样本数 {total_size}, 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}")

        # 设置随机种子以确保可复现性
        torch.manual_seed(seed)

        # 随机分割数据集
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # 为不同数据集设置不同的变换
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

        return train_size, val_size, test_size

    def create_model(self):
        """创建更复杂的CNN模型并移至GPU"""
        self.model = ComplexCNN(len(self.classes))

        # 将模型移至GPU
        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"发现{torch.cuda.device_count()}个GPU，使用DataParallel并行训练")
            self.model = nn.DataParallel(self.model)

        print(f"模型参数总数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可训练参数数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(get_gpu_memory())

        return self.model

    def setup_training(self, learning_rate=0.001, weight_decay=0.0001):
        """设置训练参数"""
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 学习率调度器 - 使用余弦退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        return self.optimizer, self.criterion, self.scheduler

    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [训练]")
        for inputs, labels in pbar:
            # 将数据移至GPU
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 更新学习率
            self.scheduler.step(epoch + inputs.size(0) / len(self.train_loader))

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({"Loss": f"{running_loss / total_samples:.4f}",
                              "Acc": f"{running_corrects.double() / total_samples:.4f}",
                              "LR": f"{current_lr:.6f}",
                              "GPU": get_gpu_memory()})

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        return epoch_loss, epoch_acc

    def validate(self):
        """在验证集上评估模型"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # 将数据移至GPU
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

        val_loss = running_loss / total_samples
        val_acc = running_corrects.double() / total_samples

        return val_loss, val_acc

    def train(self, epochs=100, early_stopping_patience=10):
        """训练模型"""
        self.epochs = epochs
        best_val_loss = float('inf')
        best_val_acc = 0.0
        early_stopping_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print(f"开始训练，共{epochs}个epoch...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练一个epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc.cpu().numpy())

            # 在验证集上评估
            val_loss, val_acc = self.validate()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc.cpu().numpy())

            # 保存最佳模型（基于验证准确率）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                early_stopping_counter = 0
                # 保存完整模型
                torch.save(self.model, self.model_save_path)
                print(f"Epoch {epoch + 1}: 保存最佳模型，验证准确率 = {best_val_acc:.4f}，验证损失 = {best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
                print(f"Epoch {epoch + 1}: 验证准确率未改善，计数 = {early_stopping_counter}/{early_stopping_patience}")

            # 早停检查
            if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                print(f"早停触发: 在{early_stopping_patience}个epoch内验证准确率未改善")
                break

            # 打印本轮结果
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} - LR: {current_lr:.6f} - "
                  f"训练: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"验证: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # 绘制训练历史
        self._plot_training_history(history)

        return history

    def _plot_training_history(self, history):
        """绘制训练历史"""
        # 添加以下两行，强制使用Agg后端
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端，用于服务器环境
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.title("trian val loss")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="train_acc")
        plt.plot(history["val_acc"], label="val_acc")
        plt.title("trian val acc")
        plt.xlabel("Epoch")
        plt.ylabel("acc")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        # 注释掉plt.show()，避免尝试显示图形界面
        # plt.show()

        print("训练历史图表已保存为 training_history.png")

    def test(self):
        """在测试集上评估模型"""
        if self.test_loader is None:
            raise ValueError("请先准备测试数据集")

        # 加载最佳模型
        self.model = torch.load(self.model_save_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"加载最佳模型用于测试")

        running_corrects = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="测试")
            for inputs, labels in pbar:
                # 将数据移至GPU
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                # 统计
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # 收集预测结果用于混淆矩阵
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({"Acc": f"{running_corrects.double() / total_samples:.4f}",
                                  "GPU": get_gpu_memory()})

        test_acc = running_corrects.double() / total_samples
        print(f"测试准确率: {test_acc:.4f}")

        # 打印混淆矩阵和分类报告
        self._print_confusion_matrix(all_labels, all_preds)

        return test_acc

    def _print_confusion_matrix(self, labels, preds):
        """打印混淆矩阵和分类报告"""
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=self.classes)

        print("\n混淆矩阵:")
        print(cm)
        print("\n分类报告:")
        print(report)

    def predict(self, image_path):
        """预测单张图像"""
        from PIL import Image

        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0).to(self.device)

        # 加载最佳模型
        self.model = torch.load(self.model_save_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 预测
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

        # 打印预测结果
        predicted_class = self.classes[preds.item()]
        confidence = conf.item() * 100

        print(f"预测结果: {predicted_class}，置信度: {confidence:.2f}%")

        # 获取每个类别的概率
        class_probs = {self.classes[i]: probs[0, i].item() * 100 for i in range(len(self.classes))}
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)

        print("\n各类别概率:")
        for cls, prob in sorted_probs:
            print(f"{cls}: {prob:.2f}%")

        return predicted_class, confidence, class_probs

    def export_to_tflite(self, onnx_path="model.onnx", tflite_path="model.tflite", quantize=False):
        """
        将PyTorch模型转换为TensorFlow Lite格式

        Args:
            onnx_path: 导出的ONNX模型路径
            tflite_path: 导出的TensorFlow Lite模型路径
            quantize: 是否进行量化
        """
        # 加载最佳模型
        self.model = torch.load(self.model_save_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 创建一个示例输入
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)

        # 导出为ONNX格式
        print(f"正在导出模型到ONNX格式: {onnx_path}")
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过")

        # 将ONNX转换为TensorFlow格式
        print(f"正在将ONNX模型转换为TensorFlow格式")
        tf_rep = prepare(onnx_model)

        # 导出为TensorFlow SavedModel格式
        saved_model_path = "saved_model"
        tf_rep.export_graph(saved_model_path)

        # 转换为TensorFlow Lite格式
        print(f"正在将TensorFlow模型转换为TensorFlow Lite格式: {tflite_path}")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

        # 应用优化（如果需要量化）
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("应用了默认优化和量化")

        # 转换模型
        tflite_model = converter.convert()

        # 保存TensorFlow Lite模型
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # 计算并打印模型大小
        original_size = os.path.getsize(self.model_save_path) / 1024 / 1024
        onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
        tflite_size = os.path.getsize(tflite_path) / 1024 / 1024

        print(f"模型转换完成:")
        print(f"  PyTorch模型大小: {original_size:.2f} MB")
        print(f"  ONNX模型大小: {onnx_size:.2f} MB")
        print(f"  TensorFlow Lite模型大小: {tflite_size:.2f} MB")

        # 保存类别标签
        labels_path = os.path.splitext(tflite_path)[0] + "_labels.txt"
        with open(labels_path, 'w') as f:
            f.write('\n'.join(self.classes))
        print(f"类别标签已保存到: {labels_path}")

        return tflite_path, labels_path


# 主函数示例
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "./data"  # 数据集路径，应包含所有类别的图像
    MODEL_SAVE_PATH = "best_rice_model.pth"
    BATCH_SIZE = 32
    EPOCHS = 50  # 增加训练轮数
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    GPU_ID = 0  # 指定要使用的GPU ID

    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"发现{torch.cuda.device_count()}个GPU:")
        for i in range(torch.cuda.device_count()):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("未发现GPU，将使用CPU运行")

    # 创建分类器实例
    classifier = RiceDiseaseClassifier(
        data_dir=DATA_DIR,
        model_save_path=MODEL_SAVE_PATH,
        batch_size=BATCH_SIZE,
        gpu_id=GPU_ID
    )

    # 准备数据（自动划分训练集、验证集和测试集）
    train_size, val_size, test_size = classifier.prepare_data(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    print(f"数据集划分完成: 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}")

    # 创建模型
    model = classifier.create_model()

    # 设置训练参数
    optimizer, criterion, scheduler = classifier.setup_training(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 训练模型
    history = classifier.train(epochs=EPOCHS)

    # 在测试集上评估模型
    test_acc = classifier.test()

    # 导出模型为TensorFlow Lite格式
    tflite_path, labels_path = classifier.export_to_tflite(
        onnx_path="rice_model.onnx",
        tflite_path="rice_model.tflite",
        quantize=True  # 设置为True启用量化，可以减小模型大小
    )
    print(f"TensorFlow Lite模型已导出到: {tflite_path}")
    print(f"类别标签已导出到: {labels_path}")

    # 预测单张图像（可选）
    # image_path = "data/class_name/image.jpg"
    # predicted_class, confidence, class_probs = classifier.predict(image_path)