import torch
from tqdm import tqdm
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np

import main

def mask2rle(img):

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def post_process(probability, threshold=0.5, min_size=100):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0
            
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def predict(model, test_loader, threshold=0.5, min_size=100):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to("cuda")
            ids = batch['id']
            
            outputs = model(images)
            outputs_sigmoid = torch.sigmoid(outputs)
            
            for i, img_id in enumerate(ids):
                probability = outputs_sigmoid[i].squeeze().cpu().numpy()
                
                mask = post_process(probability, threshold, min_size)
                
                rle = mask2rle(mask)
                results.append([img_id, rle])
    
    return pd.DataFrame(results, columns=['ImageId', 'EncodedPixels'])

def store_predictions():
    model = main.build_model()
    
    model.to("cuda")

    model.load_state_dict(torch.load(main.MODEL_PATH))

    test_dataset = main.InpaintingDataset(is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=main.BATCH_SIZE, shuffle=False, num_workers=4)

    
    model.load_state_dict(torch.load(main.MODEL_PATH))

    submission = predict(model, test_loader)
    submission.to_csv('submission.csv', index=False)
    print("Submission file created!")
    
if __name__ == "__main__":
    store_predictions()