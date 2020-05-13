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
std::fstream trace;

std::vector<int> process_csv(std::string freq_set) {
    std::vector<int> vect;
    std::stringstream ss(freq_set);
    int i;
    while (ss >> i)
    {
        vect.push_back(i);
        if (ss.peek() == ',')
            ss.ignore();
    }
    return vect;
}

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
    std::cout << "  --eps                    Sets the exploration rate" << std::endl;
    std::cout << "  --numOpponents           Sets the number of opponents" << std::endl;
    std::cout << "  --load                   If set, load weights from specified weight file" << std::endl;
    std::cout << "  --loadFile               If set, load weights from specified weight file" << std::endl;
    
    std::cout << "  --weightId               Sets the given Id for weight File" << std::endl;
    std::cout << "  --help                   Displays this help and exit" << std::endl;
    std::cout << "  --freq_set               comma separated list of frequencies" << std::endl;
}

// Returns the reward for SARSA based on current state
double getReward(hfo::status_t status) {
    double reward;
    if (status == hfo::GOAL) reward = -10;
    else if (status == hfo::CAPTURED_BY_DEFENSE) reward = 10;
    else if (status == hfo::OUT_OF_BOUNDS) reward = 10;
    else reward = 0;
    return reward;
}

// Fill state with only the required features from state_vec
void selectFeatures(int* indices, int numTMates, int numOpponents, bool oppPres) {
    int stateIndex = 0;

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
        a = hfo::MOVE;
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
                  int suffix, bool oppPres, std::vector<int> frequencies, double eps, bool load, std::string weightid,std::string loadFile) {
    // Number of features
    trace.open("trace.txt", std::fstream::out);

    int numF = oppPres ? (8 + 3 * numTMates + 2 * numOpponents) : (3 + 3 * numTMates);
    // Number of actions
    int numA = (5 + numOpponents) * frequencies.size(); //DEF_GOAL+MOVE+GTB+NOOP+RATG+MP(unum)

    // Other SARSA parameters
    // eps = 0.01;
    double discFac = 1;
    //double lambda=0.9375; THIS IS THE ACTUAL VALUE
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
                            // "_" + weightid) : "";
    loadWtFile = &s[0u];

    // std:: cout<<"loadFile"<<" "<<loadFile<<"\n"<<"loadWtFile"<<" "<<loadWtFile<<"\n";
    SarsaAgent *sa = new SarsaAgent(numF, numA, learnR, eps, lambda, fa, loadWtFile, "");

    hfo::HFOEnvironment hfo;
    hfo::status_t status;
    hfo::action_t a;
    double state[numF];
    int indices[numF];
    int action = -1;
    int action_freq = -1;
    int step = 1;
    double reward=0;
    int no_of_offense = numTMates + 1;
    selectFeatures(indices, numTMates, numOpponents, oppPres);
    hfo.connectToServer(hfo::HIGH_LEVEL_FEATURE_SET, "../HFO/bin/teams/base/config/formations-dt", port, "localhost", "base_right", false, "");
    std::queue <double> reward_queue;
    double reward_sum_2000 =0;


    for (int episode = 0; episode < (numEpi + numEpiTest); episode++) {
        reward=0;
        if ((episode + 1) % 100 == 0) {
            eps*=0.99;
            // learnR*=0.99;
            sa->update_eps(eps);
            // sa->update_learningRate(learnR);
        }

        if ((episode + 1) % 5000 == 0) {
            // Weights file
            char *wtFile;
            std::string s = "weights_" + std::to_string(suffix) +
                            "_" + weightid + "_episode_" + std::to_string(episode + 1);
            wtFile = &s[0u];
            sa -> saveWeights(wtFile);
        }
        int count = 0;
        status = hfo::IN_GAME;
        action = -1;
        action_freq = -1;
        step = 1;
        int count_steps = 0;
        double unum = -1;
        int num_steps_per_epi = 0;

        while (status == hfo::IN_GAME) {
            num_steps_per_epi++;
            if (action_freq != -1) {
                step = frequencies[action_freq];
            }
            const std::vector<float>& state_vec = hfo.getState();
            

            double Ball_X = state_vec[3], Ball_Y= state_vec[4];
            bool in_micro_region = Ball_X < -0.2 && fabs(Ball_Y)<0.5;
            bool small_step = step < 10 ; 
            double regReward = 0.1;

            // std:: cout<<"Game Started\n"<<step<<" <-- step\n";
            if (count_steps != step && action >= 0 && (a != hfo :: MARK_PLAYER ||  unum > 0)) {
                count_steps ++;
                // std:: cout<<"Micro Action start\n";  
                if (in_micro_region ^ small_step){
                    reward -= regReward;
                }
                else{
                    reward += regReward;
                }

                if (a == hfo::MARK_PLAYER) {
                    hfo.act(a, unum);
                    // std::cout << "MARKING" << unum <<"\n";
                } else {
                    hfo.act(a);
                }
                status = hfo.step();
                // std:: cout<<"Micro Action done\n";

                continue;

            } else {
                count_steps = 0;
            }

            // std:: cout<<"Macro Action done\n";


            if(action != -1 && action_freq != -1) {
                double temp = getReward(status);
                reward += temp;
                std::cout<<reward<<"<-- interim\n";
                if (episode < numEpi) {
                    sa->update(state, action, reward, discFac);
                }
            }

            // Fill up state array
            for (int i = 0; i < numF; i++) {
                state[i] = state_vec[indices[i]];
            }
            // std:: cout<<"state filled\n";
            // Get raw action
            int combined_action = sa->selectAction(state);
            // std:: cout<<"combined action "<<combined_action<<" \n";
            action = combined_action % frequencies.size();
            // std:: cout<<"action "<<action<<" \n";
            // std:: cout<<"frequencies.size() "<<frequencies.size()<<" \n";
            
            action_freq = combined_action / frequencies.size();
            // std:: cout<<"action_freq "<<action_freq<<" \n";

            // std:: cout<<"got raw action\n";
            // Get hfo::Action
            a = toAction(action, state_vec);
            // std:: cout<<"got action\n";
            if (a == hfo::MARK_PLAYER) {
                unum = state_vec[(state_vec.size() - 1 - (action - 5) * 3)];
                hfo.act(a, unum);
            } else {
                hfo.act(a);
            }
            std::string s = std::to_string(action);
            for (int state_vec_fc = 0; state_vec_fc < state_vec.size(); state_vec_fc++) {
                s += std::to_string(state_vec[state_vec_fc]) + ",";
            }
            s += "UNUM" + std::to_string(unum) + "\n";;
            status = hfo.step();
        }
        // End of episode
        if(action != -1) {
            double temp = getReward(status);
            reward += temp;
            std::cout<<reward<<"<-- eoe "<<episode<<"\n";

            if (episode < numEpi) {
                sa->update(state, action, reward, discFac);
            }
            sa->endEpisode();
            // std:: cout<<"episode_End: "<<episode<<"\n";

        }
        reward_queue.push(reward);
        reward_sum_2000 += reward;
        if(reward_queue.size()>2000){
            double reward_first =  reward_queue.front();
            reward_sum_2000 -= reward_first;
            reward_queue.pop();
        }
        trace<<"episode: "<<episode<<" , "<<"total_reward_episode: "<<reward<<" , "<<"reward: "<<reward_sum_2000/reward_queue.size()<<std::endl;



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
    double eps = 0.99;
    double lambda = 0;
    bool load = false;
    std::string weightid;

    // making two load files
    // std::string loadFile="";
    std:: string loadFile[2] = {"", ""};

    std::string freq_set = "4,32";
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
        } else if(param == "--numOpponents") {
            numOpponents = atoi(argv[++i]);
        } else if(param == "--lambda") {
            lambda = atof(argv[++i]);
        } else if(param == "--load") {
            load = true;
        } else if(param == "--weightId") {
            weightid = std::string(argv[++i]);
        } else if(param == "--freq_set") {
            freq_set = std::string(argv[++i]);
        } 
        // else if(param == "--loadFile") {
        //     loadFile = std::string(argv[++i]);
        // }
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

    std::vector<int> frequencies = process_csv(freq_set);
    int numTeammates = numOpponents - 1;
    std::thread agentThreads[numAgents];
    for (int agent = 0; agent < numAgents; agent++) {
        agentThreads[agent] = std::thread(offenseAgent, basePort,
                                          numTeammates, numOpponents, numEpisodes, numEpisodesTest, learnR, lambda,
                                          agent, opponentPresent, frequencies, eps, load, weightid,loadFile[agent]);
        sleep(5);
    }
    for (int agent = 0; agent < numAgents; agent++) {
        agentThreads[agent].join();
    }
    return 0;
}

