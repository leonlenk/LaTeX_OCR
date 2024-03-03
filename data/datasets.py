from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath('../'))
from utils.latex import crop_to_formula, renderedLaTeXLabelstr2Formula, display_formula

from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import sklearn as skl
import torch as t

class renderedLaTeXDataset(Dataset):
    def __init__(self, image_folder, lst_file, formulas_file, processor, device, cutoff = None):
        self.image_folder = image_folder
        self.lst_file = lst_file
        self.formulas_file = formulas_file
        self.train_filenames_df = pd.read_csv(self.lst_file, sep=" ", index_col = 0, header = None)
        self.formulas = open(self.formulas_file, encoding = "ISO-8859-1", newline="\n").readlines()
        self.processor = processor
        self.device = device
        self.cutoff = cutoff if cutoff else len(self.train_filenames_df)
        if cutoff is not None:
            self.train_filenames_df = self.train_filenames_df.iloc[:self.cutoff]
            self.formulas = self.formulas[:self.cutoff]
        
    def __len__(self):
        return self.cutoff
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.train_filenames_df.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert('RGBA')
        image = crop_to_formula(image)
        inputs = self.processor(images = image,  padding = "max_length", return_tensors="pt").to(self.device)
        for key in inputs:
            inputs[key] = inputs[key].squeeze() # Get rid of batch dimension since the dataloader will batch it for us.

        formula_idx = self.train_filenames_df.index[idx]
        caption = renderedLaTeXLabelstr2Formula(self.formulas[formula_idx])
        caption = self.processor.tokenizer.encode(
            caption, return_tensors="pt", padding = "max_length", max_length = 512, truncation = True, # Tweak this
            ).to(self.device).squeeze()
        
        return inputs, caption
    
def set_seed(seed):
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    skl.utils.check_random_state(seed)