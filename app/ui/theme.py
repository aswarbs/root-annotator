"""Colours, the application stylesheet and the few vector icons the UI uses (drawn with QPainter,
so there are no image files to bundle)."""
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

BG = "#F7F5F0"          # warm off-white window background
SURFACE = "#FFFFFF"
BORDER = "#E4E0D8"
TRACK = "#EFEDE7"       # segmented-control track, hover fills
TEXT = "#23292A"
MUTED = "#6B7270"
GREEN = "#2F6B4F"       # muted botanical green
GREEN_DARK = "#25563F"
GREEN_SOFT = "#E6EFE9"
ORANGE = "#E08A1E"
ORANGE_SOFT = "#FCF1E1"
ERROR = "#B3402F"
CANVAS = "#121513"

STYLESHEET = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif; }}
QMainWindow, QWidget#Root, QWidget#Page, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BG}; }}
QWidget {{ color: {TEXT}; font-size: 14px; }}
QLabel {{ background: transparent; }}
QLabel#Brand {{ font-size: 17px; font-weight: 600; }}
QLabel#H1 {{ font-size: 26px; font-weight: 600; }}
QLabel#H2 {{ font-size: 17px; font-weight: 600; }}
QLabel#Lead {{ font-size: 15px; color: {MUTED}; }}
QLabel#Mute {{ color: {MUTED}; font-size: 13px; }}
QLabel#Error {{ color: {ERROR}; font-size: 13px; }}
QLabel#Chip {{ background: {TRACK}; color: {MUTED}; border-radius: 9px; padding: 3px 10px; font-size: 12px; }}
QLabel#ChipGood {{ background: {GREEN_SOFT}; color: {GREEN}; border-radius: 9px; padding: 3px 10px; font-size: 12px; }}
QLabel#ChipNote {{ background: {ORANGE_SOFT}; color: #9A5B0C; border-radius: 9px; padding: 3px 10px; font-size: 12px; }}
QLabel#ChipWarn {{ background: #F8E4DF; color: {ERROR}; border-radius: 9px; padding: 3px 10px; font-size: 12px; }}
QLabel#Thumb {{ background: {CANVAS}; border-radius: 10px; color: #8FA397; font-size: 12px; }}
QLabel#Badge {{ background: {GREEN_SOFT}; border-radius: 28px; }}

QFrame#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#Row {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#Notice {{ background: {ORANGE_SOFT}; border: 1px solid #F0D9B4; border-radius: 10px; }}
QFrame#DropZone {{ background: #FBFAF6; border: 2px dashed #D3CEC3; border-radius: 18px; }}
QFrame#DropZone[drag="true"] {{ background: {GREEN_SOFT}; border: 2px dashed {GREEN}; }}
QFrame#Divider {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}

QPushButton {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 9px 18px; font-weight: 500; }}
QPushButton:hover {{ border-color: #C9C4B8; background: #FBFAF7; }}
QPushButton:pressed {{ background: #F1EFE9; }}
QPushButton:disabled {{ color: #A9ADA9; background: #F3F1EC; border-color: #ECE9E2; }}
QPushButton#Primary {{ background: {GREEN}; border: 1px solid {GREEN}; color: #FFFFFF; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {GREEN_DARK}; border-color: {GREEN_DARK}; }}
QPushButton#Primary:disabled {{ background: #B7CCC1; border-color: #B7CCC1; color: #F4F8F6; }}
QPushButton[size="large"] {{ padding: 13px 30px; font-size: 15px; border-radius: 12px; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {MUTED}; padding: 8px 12px; }}
QPushButton#Ghost:hover {{ color: {TEXT}; background: {TRACK}; }}
QPushButton#Ghost:disabled {{ background: transparent; color: #B9BDB9; }}
QPushButton#Danger {{ background: transparent; border: none; color: {ERROR}; padding: 8px 12px; }}
QPushButton#Danger:hover {{ background: #F8E4DF; }}
QPushButton#Icon {{ padding: 7px; border-radius: 9px; }}
QPushButton#IconFlat {{ background: transparent; border: none; padding: 6px; border-radius: 8px; }}
QPushButton#IconFlat:hover {{ background: {TRACK}; }}
QPushButton#Toggle {{ background: {SURFACE}; color: {MUTED}; padding: 7px 14px; }}
QPushButton#Toggle:checked {{ background: {GREEN_SOFT}; border: 1px solid {GREEN}; color: {GREEN}; }}

QFrame#Segmented {{ background: {TRACK}; border-radius: 11px; }}
QPushButton#SegBtn {{ background: transparent; border: none; border-radius: 9px; padding: 7px 20px;
    color: {MUTED}; font-weight: 600; }}
QPushButton#SegBtn:hover {{ color: {TEXT}; background: transparent; }}
QPushButton#SegBtn:checked {{ background: {SURFACE}; color: {TEXT}; }}

QProgressBar {{ background: #E9E6DF; border: none; border-radius: 4px; min-height: 8px; max-height: 8px; }}
QProgressBar::chunk {{ background: {GREEN}; border-radius: 4px; }}
QSlider {{ min-height: 28px; }}
QSlider::groove:horizontal {{ height: 4px; background: #DDD9D0; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {GREEN}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {SURFACE}; border: 2px solid {GREEN}; width: 12px; height: 12px;
    margin: -8px 0; border-radius: 8px; }}
QComboBox {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 14px;
    font-weight: 600; min-width: 220px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER}; selection-background-color: {GREEN_SOFT};
    selection-color: {TEXT}; outline: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #D5D1C7; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: {TEXT}; color: #FFFFFF; border: none; padding: 5px 8px; }}
"""


# ---------------------------------------------------------------- icons (24 x 24 grid)

def _poly(p, pts):
    p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))


def _path(p, ops, close=False):
    path = QPainterPath()
    path.moveTo(*ops[0])
    for pt in ops[1:]:
        path.lineTo(*pt)
    if close:
        path.closeSubpath()
    p.drawPath(path)


def _d_upload(p, c):
    p.drawLine(QPointF(12, 16), QPointF(12, 4))
    _poly(p, [(7, 9), (12, 4), (17, 9)])
    _poly(p, [(4, 15), (4, 20), (20, 20), (20, 15)])


def _d_folder(p, c):
    _path(p, [(3, 7), (9, 7), (11, 9.5), (21, 9.5), (21, 19), (3, 19)], close=True)


def _d_file(p, c):
    _path(p, [(6, 3), (14, 3), (19, 8), (19, 21), (6, 21)], close=True)
    _poly(p, [(14, 3), (14, 8), (19, 8)])


def _d_check(p, c):
    _poly(p, [(5, 12.5), (10, 17.5), (19, 7)])


def _d_close(p, c):
    p.drawLine(QPointF(6, 6), QPointF(18, 18))
    p.drawLine(QPointF(18, 6), QPointF(6, 18))


def _d_prev(p, c):
    _poly(p, [(15, 5), (8, 12), (15, 19)])


def _d_next(p, c):
    _poly(p, [(9, 5), (16, 12), (9, 19)])


def _d_arrow_right(p, c):
    p.drawLine(QPointF(4, 12), QPointF(19, 12))
    _poly(p, [(13, 6), (19, 12), (13, 18)])


def _d_undo(p, c):
    _poly(p, [(8, 5), (3.5, 10), (8, 15)])
    path = QPainterPath()
    path.moveTo(3.5, 10)
    path.lineTo(14, 10)
    path.arcTo(QRectF(9, 10, 10, 10), 90, -180)
    path.lineTo(8, 20)
    p.drawPath(path)


def _d_zoom_in(p, c):
    p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
    p.drawLine(QPointF(15.5, 15.5), QPointF(20, 20))
    p.drawLine(QPointF(7.5, 10.5), QPointF(13.5, 10.5))
    p.drawLine(QPointF(10.5, 7.5), QPointF(10.5, 13.5))


def _d_zoom_out(p, c):
    p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
    p.drawLine(QPointF(15.5, 15.5), QPointF(20, 20))
    p.drawLine(QPointF(7.5, 10.5), QPointF(13.5, 10.5))


def _d_fit(p, c):
    for pts in ([(4, 9), (4, 4), (9, 4)], [(15, 4), (20, 4), (20, 9)],
                [(20, 15), (20, 20), (15, 20)], [(9, 20), (4, 20), (4, 15)]):
        _poly(p, pts)


def _d_sparkle(p, c):
    _path(p, [(12, 3), (14, 10), (21, 12), (14, 14), (12, 21), (10, 14), (3, 12), (10, 10)], close=True)


def _d_image(p, c):
    p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2, 2)
    _poly(p, [(4, 17), (9, 12), (13, 16), (16, 13), (20, 17)])


def _d_leaf(p, c):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(c))
    path = QPainterPath()
    path.moveTo(4, 20)
    path.cubicTo(3, 11, 9, 4, 21, 3)
    path.cubicTo(21, 14, 14, 21, 4, 20)
    path.closeSubpath()
    p.drawPath(path)
    pen = QPen(QColor("#FFFFFF"), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(5.5, 18.5), QPointF(15, 9.5))


_DRAW = {
    "upload": _d_upload, "folder": _d_folder, "file": _d_file, "check": _d_check, "close": _d_close,
    "prev": _d_prev, "next": _d_next, "arrow_right": _d_arrow_right, "undo": _d_undo,
    "zoom_in": _d_zoom_in, "zoom_out": _d_zoom_out, "fit": _d_fit, "sparkle": _d_sparkle,
    "image": _d_image, "leaf": _d_leaf,
}
_CACHE: dict = {}


def icon(name: str, color: str = TEXT, size: int = 20) -> QIcon:
    """A crisp line icon of the given colour (rendered at 2x so it stays sharp on high-DPI screens)."""
    key = (name, color, size)
    if key not in _CACHE:
        px = size * 2
        pm = QPixmap(px, px)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(px / 24, px / 24)
        pen = QPen(QColor(color), 1.9)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        _DRAW[name](p, color)
        p.end()
        _CACHE[key] = QIcon(pm)
    return _CACHE[key]
