from es import *
from config import *
import argparse
from hfo import *
from model import make_model #add roll out in it maybe or take inspiration from simulate

def getReward(s):
  reward=0
  #--------------------------- 
  if s==GOAL:
    reward=-1
  #--------------------------- 
  elif s==CAPTURED_BY_DEFENSE:
    reward=+1
  #--------------------------- 
  elif s==OUT_OF_BOUNDS:
    reward=+1
  #--------------------------- 
  #Cause Unknown Do Nothing
  elif s==OUT_OF_TIME:
    reward=0
  #--------------------------- 
  elif s==IN_GAME:
    reward=0
  #--------------------------- 
  elif s==SERVER_DOWN:  
    reward=0
  #--------------------------- 
  else:
    print("Error: Unknown GameState", s)
  return reward


def purge_features(state): #check
  st=np.empty(Num_Features,dtype=np.float64)
  stateIndex=0
  tmpIndex= 9 + 3*Num_Teammates
  for i in range(len(state)):
    # Ignore first six features and teammate proximity to opponent(when opponent is absent)and opponent features
    if(i < 6 or i>9+6*Num_Teammates or (Num_Opponents==0 and ((i>9+Num_Teammates and i<=9+2*Num_Teammates) or i==9)) ): 
      continue
    #Ignore Angle and Uniform Number of Teammates
    temp =  i-tmpIndex
    if(temp > 0 and (temp % 3 == 2 or temp % 3 == 0)):
       continue
    if (i > 9+6*Num_Teammates): 
      continue
    st[stateIndex] = state[i]
    stateIndex+=1
  return st

def toAction(action):
  if action==0:
      a = MOVE
  elif action==1:
      a = REDUCE_ANGLE_TO_GOAL
  elif action==2:
      a = GO_TO_BALL
  elif action== 3:
      a = NOOP
  elif action== 4:
      a = DEFEND_GOAL
  else :
      a = MARK_PLAYER
  return a

# def rollout(agent, env):
#   obs = env.reset()
#   done = False
#   total_reward = 0
#   while not done:
#     a = agent.get_action(obs)
#     obs, reward, done = env.step(a)
#     total_reward += reward
#   return total_reward
def simulate(model,step):
  reward_list=[]
  action=-1
  count_steps=0
  unum=-1.
  reward_episode = 0
  status=IN_GAME
  while(staus==IN_GAME):
    state_vec = hfo.getState()
    if (count_steps != step and action >= 0 and (a !=  MARK_PLAYER or  unum > 0)):
      count_steps+=1
      if (a == MARK_PLAYER):
          hfo.act(a, unum)
      else:
          hfo.act(a)
      status = hfo.step()
      continue
    else:
      count_steps = 0

    if(action != -1):
      reward = getReward(status)
      reward_episode+=reward

    state = purge_features(state_vec)
    action = model.get_action(state)
    a = toAction(action)
    if (a == MARK_PLAYER):
      unum = state_vec[(state_vec.size() - 1 - (action - 5) * 3)]
      hfo.act(a, unum)
    else:
      hfo.act(a)
      count_steps+=1
      # std::string s = std::to_string(action);
      # for (int state_vec_fc = 0; state_vec_fc < state_vec.size(); state_vec_fc++) {
      #     s += std::to_string(state_vec[state_vec_fc]) + ",";
      # }
      # s += "UNUM" + std::to_string(unum) + "\n";;
      status = hfo.step()

  # End of episode
  if(action != -1):
    reward = getReward(status)
    reward_episode+=reward
  
  reward_list.append(reward_episode)
  return reward_list

def rollout(model,step,trails):
  model.env.seed(0)
  total_reward =0
  for i in range(trails):
    reward = simulate(model,step)
    total_reward+=reward[0]
  return total_reward
    


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000)
  parser.add_argument('--numAgents', type=int, default=0)
  parser.add_argument('--numTMates', type=int, default=0)

  parser.add_argument('--numOpponents', type=int, default=1)
  parser.add_argument('--numTrails',type=int,default=100)
  parser.add_argument('--numEpisodesTrain', type=int, default=500)
  parser.add_argument('--numEpisodesTest', type=int, default=2000)
  parser.add_argument('--step', type=int, default=32)
  parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
  parser.add_argument('--population', type=int, default=10, help='population')
  args=parser.parse_args()

  global Num_Features, Num_Actions, Num_Opponents,Num_Teammates
  Num_Opponents =  args.numOpponents
  Num_Teammates = args.numTMates
  Num_Features = 8+ 3*Num_Teammates + 2*Num_Opponents if [args.numOpponents > 0] else 3+3*Num_Teammates
  Num_Actions = 5+ Num_Opponents
  hfo_game = Game(env_name='hfo_game',
      input_size=Num_Features,
      outpgut_size=Num_Actions,
      time_factor=0,
      layers=[10, 0],
      activation='softmax',
      noise_bias=0.0,
      output_noise=[False, False, False],
      rnn_mode=False,
  )
  games['hfo_game'] = hfo_game
  model=make_model(hfo_game)
  num_params = model.param_count
  print("size of model", num_params)
  global hfo
  hfo = HFOEnvironment()
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET, "../HFO/bin/teams/base/config/formations-dt", args.port, "localhost", "base_right", False, "")
  cma = CMAES(num_params,sigma_init=args.sigma_init,popsize=args.population)
  es = cma

  while True:
    solutions = es.ask()
    fitlist=np.zeros(es.popsize)
    for i in range(es.popsize):
      model.set_model_params(solutions[i])
      fitlist[i]=rollout(model,args.step,args.numTrails)

    es.tell(fitlist)
    es_solution = es.result()

    model_params = es_solution[0] # best historical solution
    reward = es_solution[1] # best reward
    curr_reward = es_solution[2] # best of the current batch
    # model.set_model_params(np.array(model_params).round(4))
    print("reward",reward,"curr_reward",curr_reward)

    if reward>10: #args
      break




