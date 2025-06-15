from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer


def strip_inline_styles(widget: QWidget):
    """Recursively clear style sheets from *widget* and all its descendants."""
    if widget is None:
        return
    widget.setStyleSheet("")
    for child in widget.findChildren(QWidget):
        strip_inline_styles(child)


def start_green_police(root: QWidget, interval_ms: int = 300):
    """Start a QTimer to periodically strip inline styles from *root*.

    The timer is parented to *root* so it will be destroyed when the
    widget goes away. It returns the created QTimer instance so callers
    may stop it manually if desired.
    """
    timer = QTimer(root)
    timer.timeout.connect(lambda: strip_inline_styles(root))
    timer.start(interval_ms)
    return timer
