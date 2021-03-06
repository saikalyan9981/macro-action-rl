#include <iostream>
#include <vector>
#include <HFO.hpp>
#include <cstdlib>
#include <thread>
#include "SarsaAgent.h"
#include "CMAC.h"
#include <unistd.h>
#include <fstream>
#include <queue>

// Before running this program, first Start HFO server:
// $./bin/HFO --offense-agents numAgents




void printUsage() {
    std::cout << "Usage:123 ./high_level_sarsa_agent [Options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --numAgents <int>        Number of SARSA agents" << std::endl;
    std::cout << "                           Default: 0" << std::endl;
    std::cout << "  --numEpisodes <int>      Number of episodes to run" << std::endl;
    std::cout << "                           Default: 10" << std::endl;
    std::cout << "  --numEpisodesTest <int>  Number of episodes to test" << std::endl;
    std::cout << "                           Default: 10" << std::endl;
    std::cout << "  --basePort <int>         SARSA agent base port" << std::endl;
    std::cout << "                           Default: 6001" << std::endl;
    std::cout << "  --learnRate <float>      Learning rate of SARSA agents" << std::endl;
    std::cout << "                           Range: [0.0, 1.0]" << std::endl;
    std::cout << "                           Default: 0.1" << std::endl;
    std::cout << "  --suffix <int>           Suffix for weights files" << std::endl;
    std::cout << "                           Default: 0" << std::endl;
    std::cout << "  --noOpponent             Sets opponent present flag to false" << std::endl;
    std::cout << "  --step                   Sets the persistent step size" << std::endl;
    std::cout << "  --eps                    Sets the exploration rate" << std::endl;
    std::cout << "  --lambda                 Lambda to be used in SARSA" << std::endl;
    std::cout << "  --numOpponents           Sets the number of opponents" << std::endl;
    std::cout << "  --load                   If set, load weights from specified weight file" << std::endl;
    std::cout << "  --weightId               Sets the given Id for weight File" << std::endl;
    std::cout << "  --help                   Displays this help and exit" << std::endl;
}

// Returns the reward for SARSA based on current state
inline double getReward(hfo::status_t status) {
    double reward;
    if (status == hfo::GOAL) reward = -1;
    else if (status == hfo::CAPTURED_BY_DEFENSE) reward = 1;
    else if (status == hfo::OUT_OF_BOUNDS) reward = 1;
    else reward = 0;
    return reward;
}

// Fill state with only the required features from state_vec
// The length of the state vector is 10+6*T+3*O+2 {T is the number of opponents and O is the number of opponents}
void selectFeatures(int* indices, int numTMates, int numOpponents, bool oppPres) {

    int stateIndex = 0;

    // Features[0 - 9] - {5,8}=8
    // Features[9+T+1 - 9+2T]: teammates dists to closest opps=T
    // Features [9+3T+1 - 9+6T]: x, y, unum of teammates ignoring %3 -> unum of team mates=2T
    // Features  [9+6T+1 - 9+6T+3*O]: x, y, unum of opponents ignoring %3 -> unum of opponents=20
    // Ignored: Feature [ 9+6T+3O+1, 9+6T+3O+2]: last_action_status,stamina->2 
    // If no opponents ignore features Distance to Opponent
    // and Distance from Teammate i to Opponent are absent
    int tmpIndex = oppPres ? (9 + 3 * numTMates) : (9 + 2 * numTMates);

    int numF = 10 + 6 * numTMates + 3 * numOpponents;
    for(int i = 0; i < numF; i++) {
        // Ignore first six featues
        if(i == 5 || i == 8) continue;
        else if(i > 9 && i <= 9 + numTMates) continue; // Ignore Goal Opening angles, as invalid
        else if(i <= 9 + 3 * numTMates && i > 9 + 2 * numTMates) continue; // Ignore Pass Opening angles, as invalid
        // Ignore Uniform Number of Teammates and opponents
        int temp =  i - tmpIndex;
        if(temp > 0 && (temp % 3 == 0) )continue;
        //if (i > 9+6*numTMates) continue;
        indices[stateIndex] = i;
        stateIndex++;
    }
}

// Convert int to hfo::Action
hfo::action_t toAction(int action, const std::vector<float>& state_vec) {
    hfo::action_t a;
    switch (action) {
    case 0:
        a = hfo::INTERCEPT;
        break;
    case 1:
        a = hfo::REDUCE_ANGLE_TO_GOAL;
        break;
    case 2:
        a = hfo::GO_TO_BALL;
        break;
    case 3:
        a = hfo::NOOP;
        break;
    case 4:
        a = hfo::DEFEND_GOAL;
        break;
    default :
        a = hfo::MARK_PLAYER;
        break;
    }
    return a;
}


void offenseAgent(int port, int numTMates, int numOpponents, int numEpi, int numEpiTest, double learnR, double lambda,
                  int suffix, bool oppPres, double eps, int step, bool load, std::string weightid,std:: string loadFile) {
    std::fstream trace;
    std::string filename = "Trace" + std::to_string(suffix) ; 

    trace.open(filename, std::fstream::out);
    std::cout<<"lambda: "<<lambda<<"\n";

    // Number of features
    int numF = oppPres ? (8 + 3 * numTMates + 2 * numOpponents) : (3 + 3 * numTMates);
    // Number of actions
    // int nnumAumA = 5 + numOpponents; //DEF_GOAL+MOVE+GTB+NOOP+RATG+MP(unum)
    int numA = 5 + numOpponents; //DEF_GOAL+MOVE+GTB+NOOP+RATG+MP(unum)

    // Other SARSA parameters
    // Changed Remember testing keep it 0
    double discFac = 1;
    //double lambda=0.9375; THIS IS THE ACTUAL VALUE
    // double lambda = 0;
    // Tile coding parameter
    double resolution = 0.1;

    double range[numF];
    double min[numF];
    double res[numF];
    for(int i = 0; i < numF; i++) {
        min[i] = -1;
        range[i] = 2;
        res[i] = resolution;
    }

    CMAC *fa = new CMAC(numF, numA, range, min, res);
    char *loadWtFile;
    std::string s = loadFile;
    // load ? ("weights_" + std::to_string(suffix) +
    //                         "_" + weightid) : "";
    loadWtFile = &s[0u];
    SarsaAgent *sa = new SarsaAgent(numF, numA, learnR, eps, lambda, fa, loadWtFile, "");

    hfo::HFOEnvironment hfo;
    hfo::status_t status;
    hfo::action_t a;
    int indices[numF];
    double state[numF];
    int action = -1;
    double reward;
    double total_reward_episode;
    int no_of_offense = numTMates + 1;
    hfo.connectToServer(hfo::HIGH_LEVEL_FEATURE_SET, "../HFO/bin/teams/base/config/formations-dt", port, "localhost", "base_right", false, "");
    selectFeatures(indices, numTMates, numOpponents, oppPres);
    std::queue <double> reward_queue;
    double reward_sum_2000 =0;
    for (int episode = 0; episode < numEpi+numEpiTest; episode++) {
        if ((episode + 1) % 100 == 0) {
            eps*=0.99;
            // learnR*=0.99;
            // sa->update_eps(eps);
            // sa->update_learningRate(learnR);

        }
        // sa->update_learningRate(1./(episode+1));
        total_reward_episode = 0;
        if ((episode + 1) % 5000 == 0) {
            // Weights file
            char *wtFile;
            std::string s = "weights_" + std::to_string(suffix) +
                            "_" + weightid + "_episode_" + std::to_string(episode + 1);
            wtFile = &s[0u];
            sa -> saveWeights(wtFile);
        }
        status = hfo::IN_GAME;
        action = -1;
        int count_steps = 0;
        double unum = -1;
        int num_steps_per_epi = 0;
        while (status == hfo::IN_GAME) {
            num_steps_per_epi++;
            const std::vector<float>& state_vec = hfo.getState();

            // print ball position. 

            if (count_steps != step && action >= 0 && (a != hfo :: MARK_PLAYER ||  unum > 0)) {
                count_steps ++;
                if (a == hfo::MARK_PLAYER) {
                    hfo.act(a, unum);
                    //std::cout << "MARKING" << unum <<"\n";
                } else {
                    hfo.act(a);
                }
                status = hfo.step();
                continue;

            } else {
                count_steps = 0;
            }

            if(action != -1) {
                reward = getReward(status);
                total_reward_episode+=reward;
                if (episode < numEpi) {
                    sa->update(state, action, reward, discFac);
                }
            }

            // std::string r = "" ; 
            // Fill up state array
            for (int i = 0; i < numF; i++) {
                state[i] = state_vec[indices[i]];
                // r += std::to_string(i) + " for " + std::to_string(state[i]) + "," ; 
            }

            // trace << "State vec " << r << std::endl ; 

            // Get raw action
            action = sa->selectAction(state);

            // Get hfo::Action
            a = toAction(action, state_vec);
            if (a == hfo::MARK_PLAYER) {
                unum = state_vec[state_vec.size() -1 -2 - (action - 5) * 3];
                trace<<hfo::ActionToString(a)<< " " << unum << std::endl;
                if(unum > 0)
                {
	                hfo.act(a, unum);
                }
                else
                {
                	hfo.act(hfo::MOVE, unum);
                }
            } else {
				trace<<hfo::ActionToString(a)<<std::endl;
                hfo.act(a);
            }
            count_steps++;
            std::string s = std::to_string(action)+" <- action";
            s+= "\nSTATE Vector of Size "+std::to_string(state_vec.size())+" ";
            for (int state_vec_fc = 0; state_vec_fc < state_vec.size(); state_vec_fc++) {
                s += std::to_string(state_vec[state_vec_fc]) + ",";
            }
            s += "\nSTATE of Size "+std::to_string(numF)+" ";
            for (int i = 0; i < numF; i++) {
                s += std::to_string(state[i]) + ",";
                // state[i] = state_vec[indices[i]];
            }
            s += "\nUNUM" + std::to_string(unum) + "\n";;
            status = hfo.step();

            // trace << s << std::endl ; 

        }


        // std :: cout <<":::::::::::::" << num_steps_per_epi<< " "<<step << " "<<"\n";
        // End of episode
        if(action != -1) {
            reward = getReward(status);
            total_reward_episode+=reward;
            if (episode < numEpi) {
                sa->update(state, action, reward, discFac);
            }
            sa->endEpisode();
        }

        reward_queue.push(total_reward_episode);
        reward_sum_2000 += total_reward_episode;
        if(reward_queue.size()>2000){
            double reward_first =  reward_queue.front();
            reward_sum_2000 -= reward_first;
            reward_queue.pop();
        }
        
        // trace<<"episode: "<<episode<<" , "<<"total_reward_episode: "<<total_reward_episode<<" , "<<"reward: "<<reward_sum_2000/reward_queue.size()<<std::endl;
    }

    delete sa;
    delete fa;
}

int main(int argc, char **argv) {

    int numAgents = 0;
    int numEpisodes = 10;
    int numEpisodesTest = 10;
    int basePort = 6000;
    double learnR = 0.1;
    int suffix = 0;
    bool opponentPresent = true;
    int numOpponents = 0;
    double eps = 0.01;
    double lambda = 0;
    int step = 10;
    bool load = false;
    // Max. number of agents considered = 2. 
    std:: string loadFile[2] = {"", ""};
    std::string weightid;
    for (int i = 0; i < argc; i++) {
        std::string param = std::string(argv[i]);
        std::cout << param << "\n";
    }
    for(int i = 1; i < argc; i++) {
        std::string param = std::string(argv[i]);
        if(param == "--numAgents") {
            numAgents = atoi(argv[++i]);
        } else if(param == "--numEpisodes") {
            numEpisodes = atoi(argv[++i]);
        } else if(param == "--numEpisodesTest") {
            numEpisodesTest = atoi(argv[++i]);
        } else if(param == "--basePort") {
            basePort = atoi(argv[++i]);
        } else if(param == "--learnRate") {
            learnR = atof(argv[++i]);
            if(learnR < 0 || learnR > 1) {
                printUsage();
                return 0;
            }
        } else if(param == "--suffix") {
            suffix = atoi(argv[++i]);
        } else if(param == "--noOpponent") {
            opponentPresent = false;
        } else if(param == "--eps") {
            eps = atof(argv[++i]);
        } else if(param == "--lambda") {
            lambda = atof(argv[++i]);
        } else if(param == "--numOpponents") {
            numOpponents = atoi(argv[++i]);
        } else if(param == "--step") {
            step = atoi(argv[++i]);
        } else if(param == "--load") {
            load = true;
        } else if(param == "--weightId") {
            weightid = std::string(argv[++i]);
        }
         else if(param == "--loadFile") {
            loadFile[0] = std::string(argv[++i]);
        }
         else if(param == "--loadFile1") {
            loadFile[1] = std::string(argv[++i]);
        }
         else {
            printUsage();
            return 0;
        }
    }
    int numTeammates = numOpponents - 1;
    std::thread agentThreads[numAgents];
    for (int agent = 0; agent < numAgents; agent++) {

        agentThreads[agent] = std::thread(offenseAgent, basePort,
                                          numTeammates, numOpponents, numEpisodes, numEpisodesTest, learnR, lambda,
                                          agent, opponentPresent, eps, step, load, weightid,loadFile[agent]);
        sleep(5);
    }
    for (int agent = 0; agent < numAgents; agent++) {
        agentThreads[agent].join();
    }
    return 0;
}
