#!/bin/bash
Seed=2
Port=7168
for i in {5000..50000..5000}
# for i in {50000..50000}
do
    stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    --port $Port --no-logging --headless --deterministic --trials 52000 --seed $Seed > logs/3v3_action_space_sarsa_seed_${Seed}.log 2>&1 &
    # stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    # --port 7268 --no-logging --no-sync --deterministic --trials 52000 --seed 1 > logs/3v3_action_space_sarsa.log 2>&1 &


    PID=$!
    cd action_space_sarsa
    sleep 5

    ./action_space_sarsa --numAgents 1 --numOpponents 2 --numEpisodes 0 --numEpisodesTest 2000 --basePort $Port \
    --weightId action_space_sarsa_lambda_0.95_seed_$Seed --lambda 0.95 \
    --loadFile reg_3v3_interim/weights_0_3v3_reg_sa_action_space_sarsa_lambda_0.95_seed_${Seed}_episode_$i > ../logs/3v3_action_space_sarsa_debug_seed_${Seed}.log


    kill $PID
    sleep 5
    tail -6 ../logs/3v3_action_space_sarsa_seed_${Seed}.log > reg_3v3_interim/result_weights_0_action_space_sarsa_lambda_0.95_seed_${Seed}_episode_$i.txt

    cd ..
done

#  ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 2 --port 7168 --no-logging --headless --deterministic --trials 2000 --seed 1 > logs/helios.log 
   