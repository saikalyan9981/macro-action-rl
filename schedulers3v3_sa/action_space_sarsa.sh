#!/bin/bash

Lambda=0.9
LearnR=0.001
Step=32
Port=7260
Seed=2
TrainEpisodes=50000
TestEpisodes=2000
Eps=0.99
Freq_set=4,32

stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
--port ${Port} --no-logging --headless --deterministic --trials 52000 --seed $Seed > logs/action_space_sarsa.log 2>&1 &

PID=$!
echo $PID
cd action_space_sarsa
sleep 5

./action_space_sarsa --numAgents 1 --numOpponents 3 --numEpisodes ${TrainEpisodes} --numEpisodesTest ${TestEpisodes} --basePort ${Port} \
--weightId 3v3_reg_sa_action_space_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed} --lambda ${Lambda} --eps ${Eps} --learnRate ${LearnR} \
 --freq_set ${Freq_set} > ../logs/action_space_sarsa_debug.log

kill -SIGINT $PID
sleep 5
cd ..
