#!/bin/bash
# Step=1
for i in {5000..50000..5000}
# for i in 50000
do
    stdbuf -oL ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 1 --defense-agents 1 \
    --port 7268 --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/2v2_action_space_sarsa.log 2>&1 &
    # stdbuf -oL ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 1 --defense-agents 1 \
    # --port 7268 --no-logging --no-sync --deterministic --trials 52000 --seed 1 > logs/2v2_action_space_sarsa.log 2>&1 &


    PID=$!
    cd action_space_sarsa
    sleep 5

    ./action_space_sarsa --numAgents 1 --numOpponents 2 --numEpisodes 0 --numEpisodesTest 2000 --basePort 7268 \
    --weightId action_space_sarsa_lambda_0.95_seed_1 --lambda 0.95 --freq_set 32 \
    --loadFile 2v2/weights_0_action_space_sarsa_lambda_0.95_seed_1_episode_$i > 2v2_action_space_sarsa_debug.log


    kill $PID
    sleep 5
    tail -6 ../logs/2v2_action_space_sarsa.log > 2v2/result_weights_0_action_space_sarsa_lambda_0.95_seed_1_episode_$i.txt

    cd ..
    # break

done


#  ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 2 --port 7168 --no-logging --headless --deterministic --trials 2000 --seed 1 > logs/helios.log 
   