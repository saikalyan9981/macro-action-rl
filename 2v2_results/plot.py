import matplotlib
import os
import re
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

def get_result_dir(dirname):
    l = os.listdir(dirname)
    result={}

    for x in l:
        # print(x)
        if re.search("result", x) is None:
            continue
        y = x.split("_")
        # ind = y.index("episode")
        # print(y)
        episode = int(y[y.index("episode")+1].split(".")[0])
        goals = goals_captured(x)
        if re.search("action_space_sarsa", x):
            if "action_space_sarsa" in result:
                if episode in result["action_space_sarsa"]:
                    result["action_space_sarsa"][episode]+=[goals]
                else:
                    result["action_space_sarsa"][episode]=[]
                    result["action_space_sarsa"][episode]+=[goals]
            else:
                result["action_space_sarsa"]={}
                result["action_space_sarsa"][episode]=[]
                result["action_space_sarsa"][episode]+=[goals]
        
        if re.search("di_sarsa", x):
            if re.search("step_32",x):
                if "di_sarsa" in result:
                    # result["di_sarsa"][episode]+=[goals]

                    if episode in result["di_sarsa"]:
                        result["di_sarsa"][episode]+=[goals]
                    else:
                        result["di_sarsa"][episode]=[]
                        result["di_sarsa"][episode]+=[goals]

                else:
                    result["di_sarsa"]={}
                    result["di_sarsa"][episode]=[]
                    result["di_sarsa"][episode]+=[goals]
            if re.search("step_1",x):             
                if "sarsa" in result:
                    # result["sarsa"][episode]+=[goals]

                    if episode in result["sarsa"]:
                        result["sarsa"][episode]+=[goals]
                    else:
                        result["sarsa"][episode]=[]
                        result["sarsa"][episode]+=[goals]
                else:
                    result["sarsa"]={}
                    result["sarsa"][episode]=[]
                    result["sarsa"][episode]+=[goals]

    return result

            
print(get_result_dir("./"))
