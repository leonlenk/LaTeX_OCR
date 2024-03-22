# Hand2LaTeX: Handwriting to LaTeX OCR Project

![Hand2LaTeX](https://github.com/leonlenk/LaTeX_OCR/assets/38673735/2b66b320-af1d-4ce5-8e96-bd05b28ab052)

Welcome to Hand2LaTeX, a fun and innovative project that uses the power of PyTorch to convert your handwritten mathematical equations into LaTeX code. No more typing out complex equations! Just write it down, scan it, and let Hand2LaTeX do the rest!

## Features

- **Handwriting Recognition**: Uses a deep learning model trained on a large dataset of handwritten mathematical symbols and equations.
- **LaTeX Conversion**: Converts recognized handwriting into LaTeX code, ready to be used in your documents.
- **PyTorch Powered**: Built with PyTorch, a leading deep learning framework that allows for fast prototyping and efficient execution.

## Installation

To get started with Hand2LaTeX, you'll need to have Python 3.6+ and PyTorch 1.0+ installed. You can then clone this repository and follow the instructions below to download and preprocess the dataset.
1. `git clone Hand2LaTeX`
2. Go to [Kaggle's Handwritten Mathematical Expressions](https://www.kaggle.com/datasets/rtatman/handwritten-mathematical-expressions) and download the dataset. Move `archive.zip` into the `LaTeX_OCR` directory.
3. Run the following:
```python
conda create -n latexocr python==3.11
conda activate latexocr
pip install -r requirements.txt
bash ./setup.sh
```
You should see `all checks passed` after running `setup.sh`.
4. Images can be found in `img_data`, and image name / label pairs are in `img_data/labels.csv`.

## Todo:
- [ ] BLIP baseline.
- [ ] Handwritten text data.
- [ ] Additional rendered latex data found at https://zenodo.org/api/records/56198/files-archive.

# Not yet implemented:

## Usage

Using Hand2LaTeX is as simple as calling a function with your image file:

```python
from hand2latex import convert
latex_code = convert('path_to_your_image.png')
print(latex_code)
```

This will print the LaTeX code corresponding to the handwritten equation in the image.

## Training the Model

To train the model, you will need a dataset of handwritten mathematical symbols and equations. You can use the CROHME dataset, which is a popular choice for this task.

Once you have the dataset, you can train the model using the `train` function:

```python
from hand2latex import train

train('path_to_your_dataset')
```

This will train the model and save the trained weights to a file.

## Contributing

Contributions to Hand2LaTeX are welcome! Whether it's bug reports, feature requests, or new code, we appreciate all help. Please check out our [contributing guide](CONTRIBUTING.md) for guidelines on how to contribute.

## License

Hand2LaTeX is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

## Acknowledgements
We thank Professor Bolei Zhou as well as all the members of ACM AI. 
