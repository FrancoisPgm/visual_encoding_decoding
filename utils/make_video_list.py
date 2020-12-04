import os


video_dir = "data/friends/stimuli/s1"
out_file = "utils/video_list.txt"

video_list = [os.path.join(video_dir, p) for p in os.listdir(video_dir) if ".mkv" in p]

with open(out_file, "w") as f:
    for path in video_list:
        f.write(path+"\n")
