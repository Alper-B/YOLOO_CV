import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

import eval

device = torch.device("cuda")


IMG_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
MODEL_PATH = 'vanilla.pth'
TEST_DIR = '../kaggle_dataset/test/test'
TRAIN_DIR = '../kaggle_dataset/train/train'

IN_CHANNELS = 3
TRAIN_FROM_SCRATCH = False 


class InpaintingDataset(Dataset):
    def __init__(self, is_test=False):
        if is_test:
            root_path = TEST_DIR
        else:
            root_path = TRAIN_DIR
            self.masks = sorted([os.path.join(root_path, "masks", i) for i in os.listdir(os.path.join(root_path, "masks"))])
                    
        self.images = sorted([os.path.join(root_path, "images", i) for i in os.listdir(os.path.join(root_path, "images"))])
        
        self.overlay_edges = sorted([os.path.join(root_path, "edges", f"overlay_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.sobel_edges = sorted([os.path.join(root_path, "edges", f"sobel_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.rgb_edges = sorted([os.path.join(root_path, "edges", f"rgb_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        
        self.fourier_colored = sorted([os.path.join(root_path, "fourier", f"colored_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.fourier_filtered = sorted([os.path.join(root_path, "fourier", f"filtered_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.fourier_magnitude = sorted([os.path.join(root_path, "fourier", f"magnitude_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.fourier_phase = sorted([os.path.join(root_path, "fourier", f"phase_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        
        self.diff_pred = sorted([os.path.join(root_path, "diff", f"pred_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.diff_mask = sorted([os.path.join(root_path, "diff", f"mask_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.diff_overlay = sorted([os.path.join(root_path, "diff", f"overlay_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        self.diff_heatmap = sorted([os.path.join(root_path, "diff", f"heatmap_{i.split('.')[0]}.{i.split('.')[-1]}") for i in os.listdir(os.path.join(root_path, "images"))])
        
        
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.is_test = is_test

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img_tensor = self.transform(img)
        
        originals = Image.open(self.images[index]).convert("RGB")        
        overlay = Image.open(self.overlay_edges[index]).convert("RGB")        
        sobel = Image.open(self.sobel_edges[index]).convert("RGB")        
        rgb_edge = Image.open(self.rgb_edges[index]).convert("RGB")        
        fourier_colored = Image.open(self.fourier_colored[index]).convert("RGB")
        fourier_filtered = Image.open(self.fourier_filtered[index]).convert("RGB")
        fourier_magnitude= Image.open(self.fourier_magnitude[index]).convert("RGB")
        fourier_phase = Image.open(self.fourier_phase[index]).convert("RGB")
        diff_pred = Image.open(self.diff_pred[index]).convert("RGB")
        diff_mask = Image.open(self.diff_mask[index]).convert("RGB")
        diff_overlay = Image.open(self.diff_overlay[index]).convert("RGB")
        diff_heatmap = Image.open(self.diff_heatmap[index]).convert("RGB")
                
        originals_tensor = self.transform(originals)
        overlay_tensor = self.transform(overlay)
        sobel_tensor = self.transform(sobel)
        rgb_edge_tensor = self.transform(rgb_edge)
        fourier_colored_tensor = self.transform(fourier_colored)
        fourier_filtered_tensor = self.transform(fourier_filtered)
        fourier_magnitude_tensor = self.transform(fourier_magnitude)
        fourier_phase_tensor = self.transform(fourier_phase)
        diff_pred_tensor = self.transform(diff_pred)
        diff_mask_tensor = self.transform(diff_mask)
        diff_overlay_tensor = self.transform(diff_overlay)
        diff_heatmap_tensor = self.transform(diff_heatmap)        
        
        
        combined_img = torch.cat([img_tensor], dim=0)
        
        if self.is_test:
            img_id = os.path.basename(self.images[index]).split('.')[0]
            return {'image': combined_img, 'id': img_id}
        else:
            mask = Image.open(self.masks[index]).convert("L")
            mask_tensor = self.transform(mask)
            return {'image': combined_img, 'mask': mask_tensor, 'id': os.path.basename(self.images[index]).split('.')[0]}

    def __len__(self):
        if self.is_test:
            return len(self.images)
        else:
            return len(self.images)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def build_model():
    
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights="advprop", 
        in_channels=IN_CHANNELS,   
        classes=1,                
        activation=None,          
    )

    return model

class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs_sigmoid.sum() + targets.sum() + 1)
        
        return self.weight * bce_loss + (1 - self.weight) * dice_loss

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs):
    best_dice = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Train Loss': train_loss / (pbar.n + 1)})
        
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)
                outputs_sigmoid = torch.sigmoid(outputs)
                
                pred_masks = (outputs_sigmoid > 0.5).float()
                for i in range(masks.size(0)):
                    val_dice += dice_coef(
                        masks[i].cpu().numpy(), 
                        pred_masks[i].cpu().numpy()
                    )
        
        val_dice /= len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}")
        
        scheduler.step(val_dice)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved with Dice: {best_dice:.4f}")
            
def main():
   
    full_dataset = InpaintingDataset()
    
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = build_model()
    model.to(device)
    
    if TRAIN_FROM_SCRATCH == False:
        model.load_state_dict(torch.load(MODEL_PATH))
    
    criterion = DiceBCELoss(weight=0.5)  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, EPOCHS)

    
    eval.store_predictions()

if __name__ == "__main__":
    main()
