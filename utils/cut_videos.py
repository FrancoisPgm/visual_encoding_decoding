import cv2
import sys
import os
import numpy as np
from math import ceil

frames_dir = "data/preprocessed/frames"

with open("utils/video_list.txt", "r") as f:
        video_list = [line.rstrip('\n') for line in f]

i = int(sys.argv[1])

video_path = video_list[i]
episode = os.path.split(video_path)[1][8:-4]
episode = "s0" + episode[1:]
print(episode)

episode_dir = os.path.join(frames_dir, episode)
if not os.path.isdir(episode_dir):
    os.mkdir(episode_dir)

vid = cv2.VideoCapture(video_path)

success, image = vid.read()
count = 0
while success:
    if not count % 6:
        cv2.imwrite(os.path.join(episode_dir, "{}_frame-{}.jpg".format(episode, count//6)), image)
    count += 1
    success, image = vid.read()

print("done")

