import os


masked_list = [os.path.split(p)[1][:-3]+"nii.gz" for p in os.listdir("data/preprocessed/fmri") if ".npy" in p]
with open("utils/s1s2_wobids.txt", "r") as f:
        file_list = [line.rstrip('\n') for line in f]

remaining_list = [p for p in file_list if os.path.split(p)[1] not in masked_list]


with open("utils/remaining_files_to_mask.txt", "w") as f:
    for path in remaining_list:
        f.write(path+"\n")
