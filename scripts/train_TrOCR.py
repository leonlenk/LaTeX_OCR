import torch as t
import requests
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torchvision.transforms.v2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = t.device('cuda:{}'.format(args.gpu) if t.cuda.is_available() else 'cpu')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)

# Load LST files
import pandas as pd
import numpy as np
import re
from tqdm import tqdm, trange

import sys, os
sys.path.append(os.path.abspath('../'))
from utils.latex import crop_to_formula, renderedLaTeXLabelstr2Formula, display_formula

train_filenames_df = pd.read_csv("../rendered_LaTeX/processed_im2latex_train.lst", index_col = 0, header = None, sep = " ")
val_filenames_df = pd.read_csv("../rendered_LaTeX/processed_im2latex_val.lst", index_col = 0, header = None, sep = " ")
formulas = open("../rendered_LaTeX/im2latex_formulas.lst", encoding = "ISO-8859-1", newline="\n").readlines()

print("Number of training formulas: ", len(train_filenames_df))
print("Number of validation formulas: ", len(val_filenames_df))

max_len = max([len(formula) for formula in formulas])
print("Max length:", max_len)

from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import sklearn as skl
from data.datasets import renderedLaTeXDataset, set_seed
from data.dataset_tests import test_renderedLaTeXDataset
    
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

# Hyperparams
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
BATCH_SIZE = 4 # 10 gigs of Vram -> 4, <5 gigs of vram -> 2
SHUFFLE_DATASET = True

set_seed(0)
optimizer = t.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_transforms = transforms.Compose([
    transforms.v2.RandomAffine(degrees = 5,
                               scale = (0.7, 1.1),
                               shear = 30),
    transforms.v2.ColorJitter(brightness = 0.2,
                              contrast = 0.2,
                              saturation = 0.2,
                              hue = 0.1)
])

train_ds = renderedLaTeXDataset(image_folder = "../formula_images/", 
                                lst_file = "../rendered_LaTeX/processed_im2latex_train.lst", 
                                formulas_file = "../rendered_LaTeX/im2latex_formulas.lst", 
                                device = device,
                                processor = processor,
                                transforms = train_transforms)
val_ds = renderedLaTeXDataset(image_folder = "../formula_images/",
                                lst_file = "../rendered_LaTeX/processed_im2latex_val.lst",
                                formulas_file = "../rendered_LaTeX/im2latex_formulas.lst",
                                device = device,
                                processor = processor)
train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = SHUFFLE_DATASET, num_workers = 0)
val_dl = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)

test_renderedLaTeXDataset(train_ds, processor = processor)
test_renderedLaTeXDataset(val_ds, processor = processor)

print("Number of training samples:", len(train_ds))
print("Number of validation samples:", len(val_ds))

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.to(device)
model.train()

history = []; val_history = []; val_timesteps = []
ema_loss = None; ema_alpha = 0.95
scaler = t.cuda.amp.GradScaler(enabled = True)
for epoch in range(NUM_EPOCHS):
    with tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}") as pbar:
        for batch, captions in pbar:
            pixel_values = batch["pixel_values"]
            
            optimizer.zero_grad()
            with t.autocast(device_type = "cuda", dtype = t.float16, enabled = True):
                outputs = model(pixel_values = pixel_values,
                                labels = captions)
                loss = outputs.loss
                history.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_loss is None: ema_loss = loss.item()
            else: ema_loss = ema_loss * ema_alpha + loss.item() * (1 - ema_alpha)
            pbar.set_postfix(loss=ema_loss)
    
    model.eval()
    with t.no_grad():
        val_losses = []
        for batch, captions in tqdm(val_dl):
            pixel_values = batch["pixel_values"]
            outputs = model(pixel_values = pixel_values,
                            labels = captions)
            val_losses.append(outputs.loss.item())
        print(f"Validation loss: {np.mean(val_losses)}")
        val_history.append(np.mean(val_losses))
        val_timesteps.append(len(history) - 1)

# Save model
model.save_pretrained("../models/trocr-large-rendered-im2latex")
processor.save_pretrained("../models/trocr-large-rendered-im2latex")
t.save(history, "../models/trocr-large-rendered-im2latex/history.pt")
t.save(val_history, "../models/trocr-large-rendered-im2latex/val_history.pt")
t.save(val_timesteps, "../models/trocr-large-rendered-im2latex/val_timesteps.pt")
