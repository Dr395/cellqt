# -*- coding: utf-8 -*-
"""
细胞图像处理平台  ·  Professional Desktop Edition
================================================
►  统一视觉：Segoe UI 10 pt | 主色 #3C7ECC (Fluent 蓝) | 浅灰 #F4F4F4
►  结构层级：Ribbon-like 页签 ➜ 左侧导航 ➜ 中央工作区 ➜ 底部日志/状态
►  无任何局部 setStyleSheet ── 100 % 由一份 QSS 驱动
"""

import sys
from pathlib import Path
from PyQt5.QtCore   import Qt
from PyQt5.QtGui    import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabBar, QStackedWidget,
    QTreeWidget, QTreeWidgetItem, QSplitter, QStatusBar, QTextEdit,
    QVBoxLayout, QLabel, QPushButton
)

# ─── 业务页面占位（保持你的原逻辑） ─────────────────────────
from super_resolution       import SuperResolutionApp
from microspectra_sr        import MicrospectraSRApp
from target_detection       import TargetDetectionApp
from fluorescence_unmixing  import FluorescenceUnmixingApp
# ──────────────────────────────────────────────────────────


ACCENT   = "#3C7ECC"
HOVER    = "#2F64A1"
DISABLED = "#9FB1CC"
BG       = "#F4F4F4"
TXT      = "#202020"


# ══════════════════ 超分辨率聚合子页 ═══════════════════════
class SuperResolutionMain(QWidget):
    """左树点击时切换 0-普通超分  |  1-显微高光谱超分"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack = QStackedWidget()
        self.stack.addWidget(SuperResolutionApp())   # 0
        self.stack.addWidget(MicrospectraSRApp())    # 1
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.stack)

    def show_page(self, idx: int): self.stack.setCurrentIndex(idx)


# ═════════════════════  主窗体  ════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("细胞图像处理平台")
        self.resize(1360, 840)

        # ── Ribbon-like 页签 ──────────────────────────────
        self.ribbon = QTabBar(movable=False, tabsClosable=False)
        self.ribbon.setDocumentMode(True)             # 去掉厚边
        for name in ("工作", "调试", "系统"): self.ribbon.addTab(name)
        self.ribbon.currentChanged.connect(self._toggle_mode)

        # ── 左侧导航树 ───────────────────────────────────
        self.nav = QTreeWidget(); self.nav.setHeaderHidden(True)
        self.nav.setMaximumWidth(230); self.nav.setFont(QFont("Segoe UI", 10))
        self._populate_nav()
        self.nav.itemClicked.connect(self._nav_clicked)

        # ── 中央业务栈 ───────────────────────────────────
        self.page_sr   = SuperResolutionMain()
        self.page_det  = TargetDetectionApp()
        self.page_unm  = FluorescenceUnmixingApp()

        self.center_stack = QStackedWidget()
        for pg in (self.page_sr, self.page_det, self.page_unm):
            self.center_stack.addWidget(pg)

        # ── Splitter 布局 ────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.nav)
        splitter.addWidget(self.center_stack)
        splitter.setStretchFactor(1, 1)

        # ── 底部日志 & 状态栏 ─────────────────────────────
        self.log = QTextEdit(readOnly=True); self.log.setMaximumHeight(120)
        sb = QStatusBar()
        self.lbl_total, self.lbl_pending, self.lbl_done = (QLabel("总数量: 0"),
                                                           QLabel("未处理: 0"),
                                                           QLabel("已处理: 0"))
        for l in (self.lbl_total, self.lbl_pending, self.lbl_done):
            l.setMinimumWidth(110); sb.addPermanentWidget(l)

        # ── 垂直装配 ────────────────────────────────────
        root = QWidget(); vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)
        vbox.addWidget(self.ribbon); vbox.addWidget(splitter); vbox.addWidget(self.log)
        self.setCentralWidget(root); self.setStatusBar(sb)

        # ── 全局视觉 QSS ─────────────────────────────────
        self._apply_qss()
        self.ribbon.setCurrentIndex(0)      # 默认“工作”模式
        self.nav.setCurrentItem(self.nav.topLevelItem(0).child(0))
        self._nav_clicked(self.nav.currentItem(), 0)

        # ── 清理所有残余绿色样式 ─────────────────────────
        self._strip_inline_styles(self)

    # ░░░░░  左侧树  ░░░░░
    def _populate_nav(self):
        root_sr  = QTreeWidgetItem(self.nav, ["超分辨率重建"])
        QTreeWidgetItem(root_sr, ["普通超分辨率重建"]).setData(0, Qt.UserRole, ("sr", 0))
        QTreeWidgetItem(root_sr, ["显微高光谱超分辨率重建"]).setData(0, Qt.UserRole, ("sr", 1))

        root_det = QTreeWidgetItem(self.nav, ["目标检测"])
        root_det.setData(0, Qt.UserRole, ("page", 1))

        root_unm = QTreeWidgetItem(self.nav, ["荧光解混"])
        root_unm.setData(0, Qt.UserRole, ("page", 2))

    def _nav_clicked(self, item: QTreeWidgetItem, _col):
        data = item.data(0, Qt.UserRole)
        if not data: return
        tag, idx = data
        if tag == "page":
            self.center_stack.setCurrentIndex(idx)
        elif tag == "sr":
            self.center_stack.setCurrentIndex(0)
            self.page_sr.show_page(idx)

    # ░░░░░  Ribbon 模式开关  ░░░░░
    def _toggle_mode(self, idx: int):
        work = (idx == 0)
        self.nav.setEnabled(work); self.center_stack.setEnabled(work)
        self.log.setVisible(work)

    # ░░░░░  清空内联样式  ░░░░░
    def _strip_inline_styles(self, w: QWidget):
        if isinstance(w, QPushButton): w.setStyleSheet("")
        for c in w.findChildren(QWidget): self._strip_inline_styles(c)

    # ░░░░░  全局 QSS  ░░░░░
    def _apply_qss(self):
        qss = f"""
        QWidget            {{ background:{BG}; color:{TXT}; font:10pt "Segoe UI"; }}
        /* Ribbon Tab */
        QTabBar::tab       {{ background:{BG}; padding:6px 36px; border:1px solid transparent;
                              border-top-left-radius:4px; border-top-right-radius:4px; }}
        QTabBar::tab:selected {{ background:{ACCENT}; color:white; }}
        QTabBar::tab:hover {{ background:#D6E4F3; }}
        /* 左树 */
        QTreeWidget        {{ border-right:1px solid #C4C4C4; }}
        QTreeView::item    {{ height:28px; }}
        QTreeView::item:selected {{ background:{ACCENT}; color:white; }}
        /* Splitter 把手 */
        QSplitter::handle  {{ background:#D0D0D0; width:4px; }}
        /* 滚动条 */
        QScrollBar:vertical{{ width:8px; background:#E2E2E2; }}
        QScrollBar::handle:vertical {{ background:#B4B4B4; border-radius:4px; min-height:18px; }}
        QScrollBar::handle:vertical:hover {{ background:#929292; }}
        QScrollBar::sub-line,QScrollBar::add-line{{ height:0; }}
        /* PushButton */
        QPushButton        {{ background:{ACCENT}; color:white; border:0; border-radius:4px; height:28px; }}
        QPushButton:hover  {{ background:{HOVER}; }}
        QPushButton:disabled {{ background:{DISABLED}; }}
        /* 状态栏 & 日志 */
        QStatusBar         {{ background:#E9E9E9; }}
        QTextEdit          {{ background:white; border-top:1px solid #C6C6C6; }}
        """
        self.setStyleSheet(qss)

    # ░░░░░  业务页可调用的接口  ░░░░░
    def update_counts(self, total=0, pending=0, done=0):
        self.lbl_total.setText(f"总数量: {total}")
        self.lbl_pending.setText(f"未处理: {pending}")
        self.lbl_done.setText(f"已处理: {done}")

    def log_msg(self, msg: str): self.log.append(msg)


# ═════════════════════  入口  ══════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("windowsvista")              # 原生渲染底座
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow(); win.show()
    sys.exit(app.exec_())
