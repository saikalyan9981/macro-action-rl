#!/bin/bash
for i in {50000..50000}
do
	stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 1 --defense-agents 2 \
	--port 7160 --no-logging --no-sync --deterministic --trials 2000 --seed 2 > logs/6.log 2>&1 &

	PID=$!
	cd di_sarsa
	sleep 5

	./di_sarsa --numAgents 2 --numOpponents 3 --numEpisodes 0 --numEpisodesTest 2000 --basePort 7160 \
	--weightId di_sarsa_lambda_0.5_seed_1_episode_50000 --lambda 0.5 --step 8 --loadFile weights_0_7_episode_$i --loadFile1 weights_1_7_episode_$i > ../logs/6_debug.log

	kill $PID
    sleep 5
    tail -6 ../logs/6.log > result_6_episode_$i.txt
    cd ..


done