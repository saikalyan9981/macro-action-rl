#!/bin/bash

Lambda=0.9
LearnR=0.0001
Step=32
Port=7380
Seed=1
TrainEpisodes=50000
TestEpisodes=2000
Eps=0.99
Freq_set=32
LoadFile=3v3_sa_seed_${Seed}/weights_0_di_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed}_episode_40000


stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 1 --defense-agents 2 \
--port ${Port} --no-logging --headless --deterministic --trials 52000 --seed $Seed > logs/action_space_sarsa2.log 2>&1 &


# --headless --nosync

PID=$!
echo $PID
cd action_space_sarsa
sleep 5

# ./action_space_sarsa --numAgents 1 --numOpponents 3 --numEpisodes ${TrainEpisodes} --numEpisodesTest ${TestEpisodes} --basePort ${Port} \
# --weightId action_space_sarsa_lambda_${Lambda}_seed_$Seed --lambda ${Lambda} --eps ${Eps} --learnRate ${LearnR} \
#  --freq_set ${Freq_set} \
#   --loadFile ${LoadFile}\
#  > ../logs/action_space_sarsa_debug.log

./action_space_sarsa --numAgents 2 --numOpponents 3 --numEpisodes ${TrainEpisodes} --numEpisodesTest ${TestEpisodes} --basePort ${Port} \
--weightId action_space_sarsa_lambda_${Lambda}_seed_$Seed --lambda ${Lambda} --eps ${Eps} --learnRate ${LearnR} \
 --freq_set ${Freq_set} > ../logs/action_space_sarsa_debug2.log

kill -SIGINT $PID
sleep 5
cd ..
