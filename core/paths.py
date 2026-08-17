"""Project root resolution — works in dev and in frozen PyInstaller bundles
(sys._MEIPASS holds the extracted configs/models/assets)."""

import os
import sys

if getattr(sys, "frozen", False):
    ROOT = sys._MEIPASS
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))