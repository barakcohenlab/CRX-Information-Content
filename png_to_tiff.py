#!/usr/bin/env python3
import os
from PIL import Image

figures_dir = "Figures"
filenames = ["figure2", "supplementalFigure4"]
for f in filenames:
    img = Image.open(os.path.join(figures_dir, f + ".png")).convert("RGB")
    img.save(os.path.join(figures_dir, f + ".tiff"))
