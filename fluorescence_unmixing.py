import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QComboBox, QMessageBox, QListWidget, QListWidgetItem,
    QSizePolicy, QScrollArea, QProgressBar, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from ui_core import strip_inline_styles
# 导入DeepTrans相关模块
sys.path.append("F:\qiu_qt\DeepTrans-HSU-main")
try:
    from DeepTrans.models import create_model
    from DeepTrans.options.test_options import TestOptions
    from DeepTrans.data import create_dataset
except ImportError:
    print("警告：无法导入DeepTrans模块，请确保DeepTrans-HSU-main目录存在")

class DeletableListWidget(QListWidget):
    delete_requested = pyqtSignal(QListWidgetItem)

    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.currentItem():
            self.delete_requested.emit(self.currentItem())
        else:
            super().keyPressEvent(event)

class UnmixingThread(QThread):
    result_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, model, images, params, result_dir):
        super().__init__()
        self.model = model
        self.images = images
        self.params = params
        self.result_dir = result_dir

    def run(self):
        try:
            results = {}
            total = len(self.images)
            
            # 使用DeepTrans模型进行解混
            for i, (image_path, image) in enumerate(self.images.items()):
                try:
                    # 在这里添加DeepTrans的具体处理逻辑
                    # 这里是示例代码，需要根据实际的DeepTrans API进行修改
                    if self.model:
                        # 预处理
                        input_tensor = transforms.ToTensor()(image).unsqueeze(0)
                        
                        # 使用模型进行预测
                        with torch.no_grad():
                            output = self.model(input_tensor)
                        
                        # 后处理
                        output_image = transforms.ToPILImage()(output.squeeze(0))
                    else:
                        output_image = image.copy()
                    
                    results[os.path.basename(image_path)] = output_image
                    
                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {str(e)}")
                    continue
                
                progress = int((i + 1) / total * 100)
                self.progress_signal.emit(progress)
            
            self.result_ready.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))

class FluorescenceUnmixingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_dir = "DeepTrans-HSU-main"  # DeepTrans模型目录
        self.init_ui()
        self.loaded_images = {}
        self.unmixed_images = {}
        self.current_model = None
        strip_inline_styles(self)
        self.result_dir = "result/fluorescence_unmixing"
        os.makedirs(self.result_dir, exist_ok=True)

    def init_ui(self):
        # 设置字体
        button_font = QFont()
        button_font.setPointSize(10)

        # 主布局
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # 左侧面板
        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        model_label.setFont(button_font)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["DeepTrans-HSU"])
        self.model_combo.setFont(button_font)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        left_panel.addLayout(model_layout)

        # 参数设置
        params_layout = QVBoxLayout()
        params_label = QLabel("参数设置:")
        params_label.setFont(button_font)
        params_layout.addWidget(params_label)

        # 迭代次数
        iter_layout = QHBoxLayout()
        iter_label = QLabel("迭代次数:")
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(100, 10000)
        self.iter_spinbox.setValue(1000)
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.iter_spinbox)
        params_layout.addLayout(iter_layout)

        # 学习率
        lr_layout = QHBoxLayout()
        lr_label = QLabel("学习率:")
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 0.1)
        self.lr_spinbox.setValue(0.001)
        self.lr_spinbox.setSingleStep(0.0001)
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_spinbox)
        params_layout.addLayout(lr_layout)

        left_panel.addLayout(params_layout)

        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setFont(button_font)
        self.load_model_btn.clicked.connect(self.load_model)
        left_panel.addWidget(self.load_model_btn)

        # 图片列表
        self.image_list = DeletableListWidget()
        self.image_list.delete_requested.connect(self.delete_selected_image)
        self.image_list.itemClicked.connect(self.display_selected_image)
        left_panel.addWidget(QLabel("已加载图片:"))
        left_panel.addWidget(self.image_list)

        # 加载图片按钮
        self.load_image_btn = QPushButton("加载图片")
        self.load_image_btn.setFont(button_font)
        self.load_image_btn.clicked.connect(self.load_image)
        left_panel.addWidget(self.load_image_btn)

        # 开始解混按钮
        self.unmix_btn = QPushButton("开始解混")
        self.unmix_btn.setFont(button_font)
        self.unmix_btn.clicked.connect(self.run_unmixing)
        left_panel.addWidget(self.unmix_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        left_panel.addWidget(self.progress_bar)

        # 保存结果按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setFont(button_font)
        self.save_btn.clicked.connect(self.save_image)
        left_panel.addWidget(self.save_btn)

        # 右侧面板
        # 原始图片显示
        right_panel.addWidget(QLabel("原始图片:"))
        self.original_image = QLabel()
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setMinimumSize(400, 300)
        right_panel.addWidget(self.original_image)

        # 解混结果显示
        right_panel.addWidget(QLabel("解混结果:"))
        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setMinimumSize(400, 300)
        right_panel.addWidget(self.result_image)

        # 设置主布局
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        self.setLayout(main_layout)

    def get_unmixing_params(self):
        return {
            'iterations': self.iter_spinbox.value(),
            'learning_rate': self.lr_spinbox.value()
        }

    def load_model(self):
        try:
            # 加载DeepTrans模型
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"找不到DeepTrans模型目录: {self.model_dir}")

            # 这里添加实际的DeepTrans模型加载逻辑
            # 示例代码：
            opt = TestOptions().parse()  # 获取测试选项
            opt.model = 'deeptrans'  # 设置模型名称
            self.current_model = create_model(opt)  # 创建模型实例
            
            QMessageBox.information(self, "成功", "成功加载DeepTrans模型")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.current_model = None

    def load_image(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, '选择图片', '', 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        
        for file_path in file_paths:
            if file_path:
                try:
                    image = Image.open(file_path)
                    self.loaded_images[file_path] = image
                    item = QListWidgetItem(os.path.basename(file_path))
                    self.image_list.addItem(item)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法加载图片 {file_path}: {str(e)}")

    def run_unmixing(self):
        if not self.current_model:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if not self.loaded_images:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        params = self.get_unmixing_params()
        self.inference_thread = UnmixingThread(
            self.current_model, self.loaded_images, params, self.result_dir)
        self.inference_thread.result_ready.connect(self.on_unmixing_finished)
        self.inference_thread.error_signal.connect(self.on_unmixing_error)
        self.inference_thread.progress_signal.connect(self.update_progress)
        self.inference_thread.start()

    def on_unmixing_finished(self, results):
        self.unmixed_images.update(results)
        QMessageBox.information(self, "成功", "荧光解混完成")
        self.progress_bar.setValue(100)

    def on_unmixing_error(self, error_message):
        QMessageBox.critical(self, "错误", f"解混过程出错: {error_message}")
        self.progress_bar.setValue(0)

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def save_image(self):
        if not self.unmixed_images:
            QMessageBox.warning(self, "警告", "没有可保存的解混结果")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if save_dir:
            try:
                for image_name, image in self.unmixed_images.items():
                    save_path = os.path.join(save_dir, f"unmixed_{image_name}")
                    image.save(save_path)
                QMessageBox.information(self, "成功", "解混结果已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def display_selected_image(self, item):
        image_name = item.text()
        for path, image in self.loaded_images.items():
            if os.path.basename(path) == image_name:
                self.display_image(self.original_image, image)
                if image_name in self.unmixed_images:
                    self.display_image(self.result_image, self.unmixed_images[image_name])
                break

    def display_image(self, label, image):
        if image:
            # 转换PIL Image为QImage
            qimage = self.pil_to_qimage(image)
            
            # 调整图片大小以适应标签
            scaled_pixmap = QPixmap.fromImage(qimage).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            label.setPixmap(scaled_pixmap)

    def pil_to_qimage(self, image):
        if image.mode == "RGB":
            r, g, b = image.split()
            image = Image.merge("RGB", (b, g, r))
        elif image.mode == "RGBA":
            r, g, b, a = image.split()
            image = Image.merge("RGBA", (b, g, r, a))
        
        data = image.tobytes("raw", image.mode)
        
        if image.mode == "RGBA":
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        else:
            qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
        
        return qimage

    def delete_selected_image(self, item):
        image_name = item.text()
        row = self.image_list.row(item)
        
        # 从列表和字典中删除
        self.image_list.takeItem(row)
        
        # 从loaded_images中删除
        for path in list(self.loaded_images.keys()):
            if os.path.basename(path) == image_name:
                del self.loaded_images[path]
                break
        
        # 从unmixed_images中删除
        if image_name in self.unmixed_images:
            del self.unmixed_images[image_name]
        
        # 清除显示
        if self.image_list.count() == 0:
            self.original_image.clear()
            self.result_image.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        for file_path in files:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                try:
                    image = Image.open(file_path)
                    self.loaded_images[file_path] = image
                    item = QListWidgetItem(os.path.basename(file_path))
                    self.image_list.addItem(item)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法加载图片 {file_path}: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FluorescenceUnmixingApp()
    window.show()
    sys.exit(app.exec_()) 