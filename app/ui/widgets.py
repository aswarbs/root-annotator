"""Small reusable widgets: image viewer with zoom/pan, segmented control, step indicator,
drop zone, slice bar and a few layout helpers."""
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton, QButtonGroup, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from . import theme


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def to_pixmap(rgb: np.ndarray, height: Optional[int] = None) -> QPixmap:
    pix = QPixmap.fromImage(to_qimage(rgb))
    if height:
        pix = pix.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
    return pix


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def label(text: str = "", name: str = None, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


def button(text: str, name: str = None, icon: str = None, large: bool = False, icon_color: str = None,
           tip: str = None) -> QPushButton:
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    if large:
        b.setProperty("size", "large")
    if icon:
        color = icon_color or ("#FFFFFF" if name == "Primary" else theme.TEXT)
        b.setIcon(theme.icon(icon, color, 18))
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def icon_button(icon_name: str, tip: str, flat: bool = True) -> QPushButton:
    b = QPushButton()
    b.setObjectName("IconFlat" if flat else "Icon")
    b.setIcon(theme.icon(icon_name, theme.TEXT, 18))
    b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def card(shadow: bool = True) -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    if shadow:
        fx = QGraphicsDropShadowEffect(frame)
        fx.setBlurRadius(28)
        fx.setOffset(0, 3)
        fx.setColor(QColor(40, 40, 30, 28))
        frame.setGraphicsEffect(fx)
    return frame


class Toggle(QPushButton):
    """A pill-shaped on/off button that shows a tick while on."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setObjectName("Toggle")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggled.connect(self._sync)
        self._sync(False)

    def _sync(self, on: bool) -> None:
        self.setIcon(theme.icon("check", theme.GREEN, 16) if on else theme.icon("check", "#C9C4B8", 16))


class Segmented(QFrame):
    """Apple-style segmented control. Emits the key of the selected segment."""
    changed = pyqtSignal(str)

    def __init__(self, options):
        super().__init__()
        self.setObjectName("Segmented")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._btns = {}
        for key, text in options:
            b = QPushButton(text)
            b.setObjectName("SegBtn")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self._group.addButton(b)
            self._btns[key] = b
            lay.addWidget(b)

    def set_value(self, key: str) -> None:
        if key in self._btns:
            self._btns[key].setChecked(True)

    def set_key_visible(self, key: str, visible: bool) -> None:
        self._btns[key].setVisible(visible)


class StepButton(QAbstractButton):
    """One step of the Upload > Segment > Review > Export indicator."""

    def __init__(self, number: int, text: str):
        super().__init__()
        self.number = number
        self.setText(text)
        self.state = "todo"   # current | done | todo
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_state(self, state: str) -> None:
        self.state = state
        self.update()

    def sizeHint(self) -> QSize:
        f = QFont(self.font())
        f.setPixelSize(14)
        f.setBold(True)   # the current step is bold: size for the widest case so the label never clips
        return QSize(34 + QFontMetrics(f).horizontalAdvance(self.text()) + 16, 36)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        current, done = self.state == "current", self.state == "done"
        cy, r = self.height() / 2, 12
        circle = QRectF(2, cy - r, 2 * r, 2 * r)
        green, grey = QColor(theme.GREEN), QColor("#B9B5AA")
        if current:
            p.setBrush(green)
            p.setPen(Qt.PenStyle.NoPen)
        else:
            p.setBrush(QColor(theme.GREEN_SOFT) if done else Qt.BrushStyle.NoBrush)
            p.setPen(QPen(green if done else grey, 1.6))
        p.drawEllipse(circle)
        if done:
            pen = QPen(green, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            c = circle.center()
            path = QPainterPath()
            path.moveTo(c.x() - 4.5, c.y() + 0.5)
            path.lineTo(c.x() - 1.5, c.y() + 3.5)
            path.lineTo(c.x() + 4.5, c.y() - 3)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        else:
            p.setPen(QColor("#FFFFFF") if current else grey)
            f = QFont(self.font())
            f.setPixelSize(12)
            f.setBold(True)
            p.setFont(f)
            p.drawText(circle, Qt.AlignmentFlag.AlignCenter, str(self.number))
        f = QFont(self.font())
        f.setPixelSize(14)
        f.setBold(current)
        p.setFont(f)
        if current:
            p.setPen(QColor(theme.TEXT))
        elif enabled:
            p.setPen(QColor(theme.MUTED if not done else theme.TEXT))
        else:
            p.setPen(QColor("#B9B5AA"))
        p.drawText(QRectF(2 * r + 12, 0, self.width() - 2 * r - 12, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


class Stepper(QWidget):
    """Upload > Segment > Review > Export. Emits the index of a clicked, reachable step."""
    stepClicked = pyqtSignal(int)

    def __init__(self, names):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.buttons = []
        for i, name in enumerate(names):
            if i:
                line = QFrame()
                line.setObjectName("Divider")
                line.setFixedWidth(28)
                lay.addWidget(line, 0, Qt.AlignmentFlag.AlignVCenter)
            b = StepButton(i + 1, name)
            b.clicked.connect(lambda _=False, k=i: self.stepClicked.emit(k))
            self.buttons.append(b)
            lay.addWidget(b)

    def set_state(self, current: int, reachable) -> None:
        for i, b in enumerate(self.buttons):
            b.setEnabled(bool(reachable(i)))
            b.set_state("current" if i == current else ("done" if i < current and reachable(i) else "todo"))


class ImageViewer(QWidget):
    """The big image area: scales the picture to fit; mouse wheel zooms, drag pans when zoomed,
    double-click resets."""
    zoomChanged = pyqtSignal(float)
    _MAX_ZOOM = 12.0

    def __init__(self):
        super().__init__()
        self.setMinimumSize(380, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img: Optional[QImage] = None
        self._msg = ""
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._last: Optional[QPointF] = None

    # -- content
    def set_image(self, rgb: np.ndarray) -> None:
        self._img, self._msg = to_qimage(rgb), ""
        self.update()

    def set_message(self, text: str) -> None:
        self._img, self._msg = None, text
        self.update()

    def has_image(self) -> bool:
        return self._img is not None

    # -- view
    @property
    def zoom(self) -> float:
        return self._zoom

    def _avail(self) -> QRectF:
        return QRectF(self.rect()).adjusted(10, 10, -10, -10)

    def _target(self) -> QRectF:
        avail = self._avail()
        iw, ih = self._img.width(), self._img.height()
        s = min(avail.width() / iw, avail.height() / ih) * self._zoom
        w, h = iw * s, ih * s
        c = avail.center() + self._pan
        return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)

    def reset_view(self) -> None:
        self._zoom, self._pan = 1.0, QPointF(0, 0)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        self.zoomChanged.emit(1.0)

    def zoom_by(self, factor: float, anchor: Optional[QPointF] = None) -> None:
        if self._img is None:
            return
        old, new = self._zoom, min(self._MAX_ZOOM, max(1.0, self._zoom * factor))
        if new == old:
            return
        avail = self._avail()
        a = anchor if anchor is not None else avail.center()
        r = self._target()
        fx, fy = (a.x() - r.x()) / r.width(), (a.y() - r.y()) / r.height()
        ratio = new / old
        w2, h2 = r.width() * ratio, r.height() * ratio
        center = QPointF(a.x() - fx * w2 + w2 / 2, a.y() - fy * h2 + h2 / 2)
        self._zoom = new
        self._pan = QPointF(0, 0) if new == 1.0 else center - avail.center()
        self._clamp()
        self.setCursor(Qt.CursorShape.OpenHandCursor if new > 1.0 else Qt.CursorShape.ArrowCursor)
        self.update()
        self.zoomChanged.emit(new)

    def _clamp(self) -> None:
        if self._img is None:
            return
        r, avail = self._target(), self._avail()
        mx = max(0.0, (r.width() + avail.width()) / 2 - 60)
        my = max(0.0, (r.height() + avail.height()) / 2 - 60)
        self._pan = QPointF(max(-mx, min(mx, self._pan.x())), max(-my, min(my, self._pan.y())))

    # -- events
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 14, 14)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QColor(theme.CANVAS))
        if self._img is not None:
            p.drawImage(self._target(), self._img)
        elif self._msg:
            p.setPen(QColor("#9FB0A6"))
            f = QFont(self.font())
            f.setPixelSize(15)
            p.setFont(f)
            p.drawText(QRectF(self.rect()).adjusted(30, 30, -30, -30),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._msg)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._clamp()

    def wheelEvent(self, event) -> None:
        if self._img is not None:
            self.zoom_by(1.18 ** (event.angleDelta().y() / 120), event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.0:
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._last is not None:
            self._pan += event.position() - self._last
            self._last = event.position()
            self._clamp()
            self.update()

    def mouseReleaseEvent(self, _event) -> None:
        if self._last is not None:
            self._last = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._zoom > 1.0 else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, _event) -> None:
        self.reset_view()


class SliceBar(QWidget):
    """previous / slider / next with a "Slice 12 of 500" caption."""
    valueChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.prev_btn = icon_button("prev", "Previous slice")
        self.next_btn = icon_button("next", "Next slice")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 1)
        self.slider.valueChanged.connect(self.valueChanged)
        self.prev_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.next_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.caption = QLabel()
        self.caption.setObjectName("Mute")
        self.caption.setMinimumWidth(130)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.prev_btn)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.next_btn)
        lay.addWidget(self.caption)

    def set_range(self, maximum: int) -> None:
        self.slider.setRange(1, max(1, maximum))

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)
        self.update_caption()

    def update_caption(self) -> None:
        self.caption.setText(f"Slice {self.slider.value()} of {self.slider.maximum()}")


class DropZone(QFrame):
    """Large drag-and-drop target with the two ways of choosing files."""
    dropped = pyqtSignal(list)
    chooseFiles = pyqtSignal()
    chooseFolder = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge = QLabel()
        self.badge.setObjectName("Badge")
        self.badge.setFixedSize(56, 56)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setPixmap(theme.icon("upload", theme.GREEN, 28).pixmap(28, 28))
        self.title = QLabel("Drag and drop your scans here")
        self.title.setObjectName("H2")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub = QLabel("Volumes saved as .npy files, or a folder of slice images")
        self.sub.setObjectName("Mute")
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.files_btn = button("Choose files", "Primary", "upload")
        self.folder_btn = button("Add folder…", icon="folder")
        self.files_btn.clicked.connect(lambda: self.chooseFiles.emit())
        self.folder_btn.clicked.connect(lambda: self.chooseFolder.emit())
        row.addWidget(self.files_btn)
        row.addWidget(self.folder_btn)
        lay.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)
        lay.addSpacing(6)
        lay.addLayout(row)

    def set_compact(self, compact: bool) -> None:
        self.badge.setVisible(not compact)
        self.sub.setVisible(not compact)
        self.title.setText("Add more scans" if compact else "Drag and drop your scans here")
        self.setMinimumHeight(110 if compact else 220)

    def _drag(self, on: bool) -> None:
        self.setProperty("drag", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag(True)

    def dragLeaveEvent(self, _event) -> None:
        self._drag(False)

    def dropEvent(self, event) -> None:
        self._drag(False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.dropped.emit(paths)
            event.acceptProposedAction()
