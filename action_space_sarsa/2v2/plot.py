import matplotlib
import os
def goals_captured(fname):
    with open(fname, 'r') as fl:
        info_dict ={}
        for line in fl.readlines():
            line = line.strip().split(":")
            # print(line)
            if len(line)<2:
                continue
            # print(info_dict)
            info_dict[line[0].strip(" ")]=int(line[1])
            
        return 1 -info_dict["Goals"]/(info_dict["Defense Captured"] +info_dict["Balls Out of Bounds"] + info_dict["Goals"] + info_dict["Out of Time"])

def get_result_dir():
    os.listdir()