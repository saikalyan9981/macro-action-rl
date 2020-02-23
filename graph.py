import subprocess
import time
# subprocess.check_call("")
s = "./di_sarsa/di_sarsa --numAgents 1 --numOpponents 2 --numEpisodes 0 --numEpisodesTest 20 --basePort 7160 --weightId di_sarsa_lambda_0.5_step_32_seed_1 --lambda 0.5 --step 32 --loadFile di_sarsa/weights_0_di_sarsa_lambda_0.5_step_32_seed_1_episode_"
for i in range(1,2):
    # r = os.system("killall -9 rcssserver")
    # print(r, "adsfasdfasdadsf")
    p = subprocess.Popen("exec " + "stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 --port 7160 --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa.log 2>&1 &", shell=True)
    print(p.pid)
    cmd = s + str(i*5000)
    print(cmd)
    r = subprocess.check_call(cmd, shell=True)
    # print(r)
    time.sleep(50)
    p.kill()
    f = open("logs/di_sarsa.log")
    goals = f.readlines()[::-1][4]
    outfile = open("disarsa.txt", mode='w')
    outfile.write(goals)
    # input()