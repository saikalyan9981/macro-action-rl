

from mpi4py import MPI

import numpy as np
import json
import os
import subprocess
import sys
sys.path.insert(1,'../HFO')
from hfo import *
from model import make_model, simulate
from es import CMAES
import argparse
import time
from config import *

def make_hfo_env(port):
  global hfoe
  sprint("port",port)
  # subprocess.Popen(["stdbuf", "-oL", "../HFO/bin/HFO","--offense-npcs", "3", "--defense-npcs", "2", "--defense-agents", "1", \
  # "--port", str(port),"--no-logging", "--headless", "--deterministic" ,"--trials", "52000"])
  # ls_output=subprocess.Popen(["sleep", "100"])
  # exit()
  # , ">" , "../logs/di_sarsa_cmaes_workers"+str(port)+".log", "2>&1", "&"])
  # sprint("hfoe creation")
  hfoe = HFOEnvironment()
  hfoe.connectToServer(HIGH_LEVEL_FEATURE_SET, "../HFO/bin/teams/base/config/formations-dt", port, "localhost", "base_right", False, "")
  # sprint("leaving","make_hfo_env")

def getReward(s):
  reward=0
  #--------------------------- 
  if s==GOAL:
    reward=0
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
def simulate_hfo(model,num_episode=10):
  sprint("in simulate_hfo")

  reward_list=[]
  t_list=[]
  for i in range(num_episode):
    action=-1
    count_steps=0
    unum=-1.
    reward_episode = 0
    status=IN_GAME
    a=0
    t=0
    while(status==IN_GAME):
      t+=1
      state_vec = hfoe.getState()
      if (count_steps != step_size and action >= 0 and (a !=  MARK_PLAYER or  unum > 0)):
        count_steps+=1
        if (a == MARK_PLAYER):
            hfoe.act(a, unum)
        else:
            hfoe.act(a)
        status = hfoe.step()
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
        unum = state_vec[(len(state_vec) - 1 - (action - 5) * 3)]
        hfoe.act(a, unum)
      else:
        hfoe.act(a)
        count_steps+=1
        # std::string s = std::to_string(action);
        # for (int state_vec_fc = 0; state_vec_fc < state_vec.size(); state_vec_fc++) {
        #     s += std::to_string(state_vec[state_vec_fc]) + ",";
        # }
        # s += "UNUM" + std::to_string(unum) + "\n";;
        status = hfoe.step()

    # End of episode
    if(action != -1):
      reward = getReward(status)
      reward_episode+=reward
    
    reward_list.append(reward_episode)
    t_list.append(t)
  return reward_list,t_list

### ES related code
num_episode = 1
eval_steps = 25 # evaluate every N_eval steps
retrain_mode = True
cap_time_mode = True

num_worker = 8
num_worker_trial = 16

population = num_worker * num_worker_trial

gamename = 'invalid_gamename'
optimizer = 'pepg'
antithetic = True
batch_mode = 'mean'

# seed for reproducibility
seed_start = 0
### name of the file (can override):
filebase = None

game = None
model = None
num_params = -1

es = None

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PRECISION = 10000
SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
RESULT_PACKET_SIZE = 4*num_worker_trial
###

def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
  global population, filebase, game, model, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE

  population = num_worker * num_worker_trial
  filebase = 'log/'+gamename+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  hfo_game = Game(env_name='hfo_game',
    input_size=Num_Features,
    output_size=Num_Actions,
    time_factor=0,
    layers=[10, 10],
    activation='softmax',
    noise_bias=0.0,
    output_noise=[False, False, False],
    rnn_mode=False,
  )
  game=hfo_game
  model = make_model(game)
  num_params = model.param_count
  print("size of model", num_params)
  print("optimizer",optimizer)
  
  if optimizer == 'ses':
    ses = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.2,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ses
  elif optimizer == 'ga':
    ga = SimpleGA(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ga
  elif optimizer == 'cma':
    cma = CMAES(num_params,
      sigma_init=sigma_init,
      popsize=population)
    es = cma
  elif optimizer == 'pepg':
    pepg = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.20,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      weight_decay=0.005,
      popsize=population)
    es = pepg
  else:
    oes = OpenES(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      antithetic=antithetic,
      weight_decay=0.005,
      popsize=population)
    es = oes

  PRECISION = 10000
  SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
  RESULT_PACKET_SIZE = 4*num_worker_trial
###

def sprint(*args):
  print(args,file=file1,flush=True) # if python3, can do print(*args)
  # print(args)
  # sys.stdout.flush()

class OldSeeder:
  def __init__(self, init_seed=0):
    self._seed = init_seed
  def next_seed(self):
    result = self._seed
    self._seed += 1
    return result
  def next_batch(self, batch_size):
    result = np.arange(self._seed, self._seed+batch_size).tolist()
    self._seed += batch_size
    return result

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
  n = len(seeds)
  result = []
  worker_num = 0
  for i in range(n):
    worker_num = int(i / num_worker_trial) + 1
    result.append([worker_num, i, seeds[i], train_mode, max_len])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))
  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)
  return result

def decode_solution_packet(packet):
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float)/PRECISION])
  return result

def encode_result_packet(results):
  r = np.array(results)
  r[:, 2:4] *= PRECISION
  return r.flatten().astype(np.int32)

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, 4)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  fits = r[:, 2].astype(np.float)/PRECISION
  fits = fits.tolist()
  times = r[:, 3].astype(np.float)/PRECISION
  times = times.tolist()
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], fits[i], times[i]])
  return result

def worker(weights, seed, train_mode_int=1, max_len=-1):
  sprint("in worker")
  train_mode = (train_mode_int == 1)
  model.set_model_params(weights)
  reward_list, t_list = simulate_hfo(model,num_episode=num_episode)
  if batch_mode == 'min':
    reward = np.min(reward_list)
  else:
    reward = np.mean(reward_list)
  t = np.mean(t_list)
  return reward, t

def slave():
  # port
  sprint("in slave")

  port=current_port.next_seed()+10*rank
  make_hfo_env(port)
  # model.make_env()
  packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
  t=0
  while 1:
    t+=1
    comm.Recv(packet, source=0)
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    solutions = decode_solution_packet(packet)
    # sprint("in decode_solution_packet",solutions)

    results = []
    for solution in solutions:
      worker_id, jobidx, seed, train_mode, max_len, weights = solution
      assert (train_mode == 1 or train_mode == 0), str(train_mode)
      worker_id = int(worker_id)
      possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
      assert worker_id == rank, possible_error
      jobidx = int(jobidx)
      seed = int(seed)
      fitness, timesteps = worker(weights, seed, train_mode, max_len)
      results.append([worker_id, jobidx, fitness, timesteps])
    result_packet = encode_result_packet(results)
    assert len(result_packet) == RESULT_PACKET_SIZE
    comm.Send(result_packet, dest=0)

def send_packets_to_slaves(packet_list):
  num_worker = comm.Get_size()
  # print("packet_list",packet_list,"num_worker",num_worker)
  assert len(packet_list) == num_worker-1
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    comm.Send(packet, dest=i)

def receive_packets_from_slaves():
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

  reward_list_total = np.zeros((population, 2))

  check_results = np.ones(population, dtype=np.int)
  for i in range(1, num_worker+1):
    comm.Recv(result_packet, source=i)
    results = decode_result_packet(result_packet)
    for result in results:
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, 0] = result[2]
      reward_list_total[idx, 1] = result[3]
      check_results[idx] = 0

  check_sum = check_results.sum()
  assert check_sum == 0, check_sum
  return reward_list_total

def evaluate_batch(model_params, max_len=-1):
  # duplicate model_params
  solutions = []
  for i in range(es.popsize):
    solutions.append(np.copy(model_params))

  seeds = np.arange(es.popsize)

  packet_list = encode_solution_packets(seeds, solutions, train_mode=0, max_len=max_len)

  send_packets_to_slaves(packet_list)
  reward_list_total = receive_packets_from_slaves()

  reward_list = reward_list_total[:, 0] # get rewards
  return np.mean(reward_list)

def master():
  sprint("in master")
  start_time = int(time.time())
  sprint("training", gamename)
  sprint("population", es.popsize)
  sprint("num_worker", num_worker)
  sprint("num_worker_trial", num_worker_trial)
  sys.stdout.flush()

  seeder = Seeder(seed_start)
  

  filename = filebase+'.json'
  filename_log = filebase+'.log.json'
  filename_hist = filebase+'.hist.json'
  filename_best = filebase+'.best.json'

  # model.make_env()
  # port=current_port.next_seed()+rank
  # make_hfo_env(port)


  t = 0

  history = []
  eval_log = []
  best_reward_eval = 0
  best_model_params_eval = None

  max_len = -1 # max time steps (-1 means ignore)

  while t<total_steps:
    t += 1

    solutions = es.ask()

    if antithetic:
      seeds = seeder.next_batch(int(es.popsize/2))
      seeds = seeds+seeds
    else:
      seeds = seeder.next_batch(es.popsize)

    packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

    send_packets_to_slaves(packet_list)
    reward_list_total = receive_packets_from_slaves()

    reward_list = reward_list_total[:, 0] # get rewards

    mean_time_step = int(np.mean(reward_list_total[:, 1])*100)/100. # get average time step
    max_time_step = int(np.max(reward_list_total[:, 1])*100)/100. # get average time step
    avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
    std_reward = int(np.std(reward_list)*100)/100. # get average time step

    es.tell(reward_list)

    es_solution = es.result()
    model_params = es_solution[0] # best historical solution
    reward = es_solution[1] # best reward
    curr_reward = es_solution[2] # best of the current batch
    model.set_model_params(np.array(model_params).round(4))
    sprint("iteration",t,"reward",reward,"reward_list",reward_list)
    r_max = int(np.max(reward_list)*100)/100.
    r_min = int(np.min(reward_list)*100)/100.

    curr_time = int(time.time()) - start_time

    h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

    if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)
    else:
      max_len = -1

    history.append(h)

    with open(filename, 'wt') as out:
      res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

    with open(filename_hist, 'wt') as out:
      res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    sprint(gamename, h)

    if (t == 1):
      best_reward_eval = avg_reward
    if (t % eval_steps == 0): # evaluate on actual task at hand

      prev_best_reward_eval = best_reward_eval
      model_params_quantized = np.array(es.current_param()).round(4)
      reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
      model_params_quantized = model_params_quantized.tolist()
      improvement = reward_eval - best_reward_eval
      eval_log.append([t, reward_eval, model_params_quantized])
      with open(filename_log, 'wt') as out:
        res = json.dump(eval_log, out)
      if (len(eval_log) == 1 or reward_eval > best_reward_eval):
        best_reward_eval = reward_eval
        best_model_params_eval = model_params_quantized
      else:
        if retrain_mode:
          sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
          es.set_mu(best_model_params_eval)
      with open(filename_best, 'wt') as out:
        res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
      sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


def main(args):

  global gamename, optimizer, num_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode,current_port
  global Num_Features, Num_Actions, Num_Opponents,Num_Teammates,step_size,total_steps
  
  gamename = args.gamename
  optimizer = args.optimizer
  num_episode = args.num_episode
  eval_steps = args.eval_steps
  num_worker = args.num_worker
  num_worker_trial = args.num_worker_trial
  antithetic = (args.antithetic == 1)
  retrain_mode = (args.retrain == 1)
  cap_time_mode= (args.cap_time == 1)
  seed_start = args.seed_start
  current_port=OldSeeder(args.port_start)
  step_size = args.step
  total_steps=args.total_steps

  Num_Opponents =  args.numOpponents
  Num_Teammates = args.numTMates
  Num_Features = 8+ 3*Num_Teammates + 2*Num_Opponents if [args.numOpponents > 0] else 3+3*Num_Teammates
  Num_Actions = 5+ Num_Opponents

  initialize_settings(args.sigma_init, args.sigma_decay)

  sprint("process", rank, "out of total ", comm.Get_size(), "started")
  if (rank == 0):
    master()
  else:
    slave()

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers',rank, nworkers)
    return "child"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train policy on HFO environment '
                                                'using  cma'))

  parser = argparse.ArgumentParser()
  parser.add_argument('--gamename', type=str, help='hfo_game etc.',default="hfo_game")

  parser.add_argument('--port_start', type=int, default=8000)
  parser.add_argument('--numAgents', type=int, default=1)
  parser.add_argument('--numTMates', type=int, default=2)

  parser.add_argument('--numOpponents', type=int, default=3)
  parser.add_argument('--numTrails',type=int,default=300)
  # parser.add_argument('--numEpisodesTrain', type=int, default=500)
  # parser.add_argument('--numEpisodesTest', type=int, default=2000)
  parser.add_argument('--step', type=int, default=32)
  parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
  parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
  parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')
  parser.add_argument('--threshold', type=float, default=0.7,help='threshold')
  parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
  parser.add_argument('-e', '--num_episode', type=int, default=300, help='num episodes per trial')
  parser.add_argument('--eval_steps', type=int, default=25, help='evaluate every eval_steps step')
  parser.add_argument('--total_steps', type=int, default=100, help='max steps')
  parser.add_argument('-n', '--num_worker', type=int, default=8)
  parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker', default=4)
  parser.add_argument('--antithetic', type=int, default=1, help='set to 0 to disable antithetic sampling')
  parser.add_argument('--cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
  parser.add_argument('--retrain', type=int, default=0, help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
  
  args=parser.parse_args()

# global hfo
  # print(args)
  # exit()
  global file1
  file1=open("reward_logs.txt","w")

  # args = parser.parse_args()
  if "parent" == mpi_fork(args.num_worker+1): sys.exit()
  main(args)
