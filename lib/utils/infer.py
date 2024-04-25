import cv2
import pandas as pd
from tqdm.notebook import tqdm
from moviepy.editor import VideoFileClip

def dump_quantization(
    df: pd.DataFrame,
    CODEBOOK: str,
    CODE_ID: int,
    video_path: str,      
    quantization_path: str
):
    for rec in tqdm(df[df[CODEBOOK] == CODE_ID].to_dict(orient='records')):
        # save frame video to disk
        video = rec['vocab']
        video_path = f"{video_path}/{video}.mp4"
        start_idx = rec['start_idx']
        end_idx = rec['end_idx']
        label = rec['Code_1']

        cap = cv2.VideoCapture(video_path)
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
