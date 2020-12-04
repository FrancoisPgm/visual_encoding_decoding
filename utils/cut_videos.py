import cv2
import sys
import os
import numpy as np
from math import ceil

TR = 1.49
DELAY = 5.4

with open("utils/video_list.txt", "r") as f:
        video_list = [line.rstrip('\n') for line in f]

i = int(sys.argv[1])

video_path = video_list[i]
episode = os.path.split(video_path)[1][8:-4]
with open("utils/sub-01_preproc_list.txt", "r") as f:
        preproc_path = [line.rstrip('\n') for line in f if episode in f][0]

n_volumes = np.load(preproc_path).shape[0]
frames = []

vid = cv2.VideoCapture(video_path)
fps = vid.get(cv2.CAP_PROP_FPS)

sucess, image = vid.read()
while success:
    frames.append(image)
    sucess, image = vid.read()

num_frames = len(frames)

for vol_num in range(n_volumes):
    frame_num = int(vol_num*TR*fps) - ceil(DELAY/fps)
    if frame_num > 0:
        cv2.imwrite("data/preprocessed/frames/{}_t-{}.jpg".format(episode, vol_num), frames[frame_num])

print("done", episode)

