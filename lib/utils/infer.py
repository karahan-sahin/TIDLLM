import cv2
import torch
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from moviepy.editor import VideoFileClip

def get_quantization(model, dataset):

    model.eval()
    dataloader = DataLoader(
        dataset, 
        batch_size=10, 
        shuffle=True,
        collate_fn=dataset.collate_fn   
    )

    dfs = []
    for train_sample in tqdm(dataloader):
        with torch.no_grad():
            quantized, indices, commitment_loss = model(train_sample['array'].float())
            quant = {
                'vocab': train_sample['tokens'],
                'start_idx': train_sample['start_idx'],
                'end_idx': train_sample['end_idx']
            }

            for index in range(indices.shape[1]):
                quant[f'Code_{index}'] = indices[:, index].cpu().numpy()

            dfs.append(pd.DataFrame(quant))

    df = pd.concat(dfs)
    df.start_idx = df.start_idx.astype(int)
    df.end_idx = df.end_idx.astype(int)

    return df

    

def dump_quantization(
    df: pd.DataFrame,
    video_path: str,      
    num_quantizers: int, 
    quantization_path: str
):
    for rec in tqdm(df.to_dict(orient='records')):
        # save frame video to disk
        video = rec['vocab']
        v_path = f"{video_path}/{video}.mp4"
        start_idx = rec['start_idx']
        end_idx = rec['end_idx']
        label = '_'.join([ str(rec[f'Code_{i}']) for i in range(num_quantizers) ])

        cap = cv2.VideoCapture(v_path)
        ret, frame = cap.read()
        
        import os
        if not os.path.exists(f'{quantization_path}/{label}'):
            os.mkdir(f'{quantization_path}/{label}')

        FRAMES = []
        for i in range(end_idx+1):
            ret, frame = cap.read()
            if i >= start_idx and i < end_idx:
                FRAMES.append(frame)

        # write frames to video
        out = cv2.VideoWriter(f'{quantization_path}/{label}/{video}_{start_idx}_{end_idx}.avi', 
                              cv2.VideoWriter_fourcc(*'DIVX'), 
                              15, 
                              (frame.shape[1], frame.shape[0]))
        for frame in FRAMES:
            out.write(frame)

        out.release()      

        videoClip = VideoFileClip(f"{quantization_path}/{label}/{video}_{start_idx}_{end_idx}.avi")
        videoClip.write_gif(f"{quantization_path}/{label}/{video}_{start_idx}_{end_idx}.gif")

        os.remove(f"{quantization_path}/{label}/{video}_{start_idx}_{end_idx}.avi")
