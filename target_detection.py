import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QFrame, QGridLayout, QScrollArea, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import glob
import datetime
from ui_core import strip_inline_styles
class TargetDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("目标检测（YOLO）")
        self.setMinimumSize(1200, 800)
        self.model_path = None
        self.model = None
        self.image_paths = []
        self.result_image_paths = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        button_font = QFont()
        button_font.setPointSize(10)

        # 模型选择分组
        model_group = QFrame()
        model_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        model_layout = QVBoxLayout(model_group)
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.setFont(button_font)
        self.model_path_label = QLabel("未选择模型文件")
        self.model_path_label.setWordWrap(True)
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setFont(button_font)
        model_layout.addWidget(self.select_model_btn)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_btn)
        left_layout.addWidget(model_group)

        # 图片选择分组
        image_group = QFrame()
        image_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        image_layout = QVBoxLayout(image_group)
        self.select_image_btn = QPushButton("选择图片/文件夹")
        self.select_image_btn.setFont(button_font)
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(200)
        image_layout.addWidget(self.select_image_btn)
        image_layout.addWidget(self.image_list)
        left_layout.addWidget(image_group)

        # 检测与保存分组
        action_group = QFrame()
        action_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        action_layout = QVBoxLayout(action_group)
        self.run_detection_btn = QPushButton("开始检测")
        self.run_detection_btn.setFont(button_font)
        self.save_result_btn = QPushButton("保存检测结果")
        self.save_result_btn.setFont(button_font)
        action_layout.addWidget(self.run_detection_btn)
        action_layout.addWidget(self.save_result_btn)
        left_layout.addWidget(action_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        main_layout.addWidget(left_panel)

        # 右侧结果显示区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.grid_layout = QGridLayout(scroll_content)
        self.grid_layout.setSpacing(10)
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)
        main_layout.addWidget(right_panel)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 4)

        # 信号连接
        self.select_model_btn.clicked.connect(self.select_model_file)
        self.load_model_btn.clicked.connect(self.load_model)
        self.select_image_btn.clicked.connect(self.select_images)
        self.run_detection_btn.clicked.connect(self.run_detection)
        self.save_result_btn.clicked.connect(self.save_results)
        self.load_model_btn.setEnabled(False)
        self.select_image_btn.setEnabled(False)
        self.run_detection_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)

    def clear_results(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def add_result_image(self, image_path, title):
        frame = QFrame()
        frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout = QVBoxLayout(frame)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(200, 200)
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        layout.addWidget(image_label)
        row = self.grid_layout.count() // 4
        col = self.grid_layout.count() % 4
        self.grid_layout.addWidget(frame, row, col)

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "./model", "YOLO模型文件 (*.pt);;所有文件 (*.*)")
        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(f"已选择: {os.path.basename(file_path)}")
            self.load_model_btn.setEnabled(True)

    def load_model(self):
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.model = YOLO(self.model_path)
            self.progress_bar.setValue(100)
            self.select_image_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def select_images(self):
        mode, ok = QInputDialog.getItem(self, "选择模式", "请选择加载方式：", ["图片", "文件夹"], 0, False)
        if not ok:
            return
        files = []
        if mode == "图片":
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "./data", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        elif mode == "文件夹":
            folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "./data")
            if folder:
                files = glob.glob(os.path.join(folder, "*.png")) + \
                        glob.glob(os.path.join(folder, "*.jpg")) + \
                        glob.glob(os.path.join(folder, "*.jpeg")) + \
                        glob.glob(os.path.join(folder, "*.bmp"))
        if files:
            self.image_paths = files
            self.image_list.clear()
            for file_path in self.image_paths:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.image_list.addItem(item)
            if self.model:
                self.run_detection_btn.setEnabled(True)

    def run_detection(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先选择图片")
            return
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            # 检测结果保存目录
            save_dir = os.path.abspath("./detect_results")
            os.makedirs(save_dir, exist_ok=True)
            self.result_image_paths = []
            for idx, img_path in enumerate(self.image_paths):
                results = self.model.predict(source=img_path, save=True, save_txt=True, conf=0.1, project=save_dir, name="yolo_results", exist_ok=True)
                # ultralytics会在save_dir/yolo_results/下生成检测图片
                # 获取最新生成的图片
                base_name = os.path.basename(img_path)
                result_img = os.path.join(save_dir, "yolo_results", base_name)
                if os.path.exists(result_img):
                    self.result_image_paths.append(result_img)
                self.progress_bar.setValue(int((idx + 1) / len(self.image_paths) * 100))
            self.clear_results()
            for i, img_path in enumerate(self.result_image_paths):
                self.add_result_image(img_path, f"检测结果 {i+1}")
            self.save_result_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "检测完成，结果已显示")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def save_results(self):
        if not self.result_image_paths:
            QMessageBox.warning(self, "警告", "没有检测结果可保存")
            return
        # 统一保存到 ./result/target_detection/当前批次/
        batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(os.getcwd(), "result", "target_detection", batch_time)
        os.makedirs(save_root, exist_ok=True)
        save_folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹", save_root)
        if save_folder:
            for idx, img_path in enumerate(self.result_image_paths):
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)
                target_path = os.path.join(save_folder, f"{name}.png")
                count = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(save_folder, f"{name}_{count}.png")
                    count += 1
                try:
                    with open(img_path, 'rb') as fsrc, open(target_path, 'wb') as fdst:
                        fdst.write(fsrc.read())
                    # 同步保存txt（如果有）
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        txt_target = os.path.join(save_folder, os.path.basename(txt_path))
                        count_txt = 1
                        while os.path.exists(txt_target):
                            txt_target = os.path.join(save_folder, f"{name}_{count_txt}.txt")
                            count_txt += 1
                        with open(txt_path, 'rb') as fsrc, open(txt_target, 'wb') as fdst:
                            fdst.write(fsrc.read())
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"保存 {base_name} 失败: {str(e)}")
            QMessageBox.information(self, "成功", "检测结果已保存")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TargetDetectionApp()
    window.show()
    sys.exit(app.exec_())