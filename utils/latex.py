from PIL import Image
import re
import numpy as np
from IPython.display import display, Math, Latex

def crop_to_formula(image, padding = 30):
    # Image: 4 channel image with alpha.
    # Convert black pixels to white pixels.
    data = np.array(image)
    red, green, blue, alpha = data.T
    black_areas = (red < 10) & (blue < 10) & (green < 10)
    # Convert alpha to white.
    data[..., -1] = 255
    # Crop a box around the area that contains black pixels.
    coords = np.argwhere(black_areas)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    # Add padding.
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)
    image = Image.fromarray(data[y0:y1, x0:x1])
    return image.convert('RGB')

def renderedLaTeXLabelstr2Formula(label: str):
    # We're matching \\label{...whatever} and removing it
    label = re.sub(r"\\label\{[^\}]*\}", "", label)
    # We match \, and remove it.
    label = re.sub(r"\\,", "", label)
    return label

def display_formula(latex: str):
    # Remove \mbox{...} - not supported by the inline MathJax renderer
    parsed_latex = re.sub(r"\\mbox\{[^\}]*\}", "", latex)
    display(Math(parsed_latex))