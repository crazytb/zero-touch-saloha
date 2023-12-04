import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from numpy.random import default_rng

# Hyperparameters
learning_rate = 0.0001
gamma = 1

# Parameters
BUFFERSIZE = 20  # Def. 20
NUMNODES = 20
DIMSTATES = 2 * NUMNODES + 1
FRAMETIME = 270  # microseconds
TIMEEPOCH = 300  # microseconds
FRAMETXSLOT = 30
FRAMEAGGLIMIT = int(TIMEEPOCH/FRAMETXSLOT)
BEACONINTERVAL = 100_000  # microseconds
# MAXAOI = int(np.ceil(BEACONINTERVAL / TIMEEPOCH))
ACCESSPROB = 1 / NUMNODES
# ACCESSPROB = 1
POWERCOEFF = 0.1
AOIPENALTY = 1
PER = 0.1
PEAKAOITHRES = 20_000   # That is, 5 000 for 5ms, (5,20)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, round(n_observations/2))
        self.layer2 = nn.Linear(round(n_observations/2), round(n_observations/2))
        self.layer3 = nn.Linear(round(n_observations/2), n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ShowerEnv(Env):
    def __init__(self):
        super(ShowerEnv, self).__init__()
        # Actions we can take FORWARD, DISCARD, and SKIP
        self.action_space = Discrete(3)
        self.max_channel_quality = 2
        self.max_current_time = 0
        self.max_current_aoi = 1
        self.max_buffer_location = 1
        
        self.observation_space = spaces.Dict({
            "channel_quality": spaces.Discrete(self.max_channel_quality),
            "current_time": spaces.Box(low=0, high=1, shape=(1, 1)),
            "current_aois": spaces.Box(low=0, high=1, shape=(1, NUMNODES)),
            "node_location": spaces.MultiDiscrete([BUFFERSIZE] * NUMNODES),
            "node_aoi": spaces.Box(low=0, high=1, shape=(1, NUMNODES)),
        })
        
        self.inbuffer_info_node = np.zeros([BUFFERSIZE], dtype=int)
        self.inbuffer_info_timestamp = np.zeros([BUFFERSIZE], dtype=float)
        self.rng = default_rng()
        self.current_obs = None

    def _get_node_info(self, buffer_node_info, buffer_node_timestamp):
        node_location = BUFFERSIZE * np.ones([NUMNODES], dtype=int)
        node_aoi = np.zeros([NUMNODES], dtype=float)
        for node_i in range(1, NUMNODES + 1):
            try:
                location = np.where(buffer_node_info == node_i)[0][0]
                node_location[node_i - 1] = location
                node_aoi[node_i - 1] = buffer_node_timestamp[location] / BEACONINTERVAL
            except:
                pass
        return node_location, node_aoi

    def _stepfunc(self, thres, x):
        if x > thres:
            return 1
        else:
            return 0
    
    def _get_obs(self):
        return {
            "channel_quality": self.channel_quality,
            "current_time": self.current_time,
            "current_aois": self.current_aois,
            "node_location": self.node_location,
            "node_aoi": self.node_aoi,
        }

    def _fill_first_zero(self, arr1, arr2):
        if not np.any(arr1 == 0):
            return arr1  # No zeros in arr1, return it as is

        zero_index = np.where(arr1 == 0)[0][0]
        remaining_zeros = np.count_nonzero(arr1 == 0) - zero_index  # Calculate the number of remaining zeros after the first zero
        
        if remaining_zeros >= len(arr2):
            arr1[0][zero_index:zero_index + len(arr2)] = arr2[:remaining_zeros]
        else:
            arr1[0][zero_index:zero_index + remaining_zeros] = arr2[:remaining_zeros]

        return arr1

    def _flatten_dict_values(self, dict):
        flattened = np.array([])
        for v in list(dict.values()):
            if isinstance(v, np.ndarray):
                flattened = np.concatenate([flattened, np.squeeze(np.reshape(v, [1, v.size]))])
            else:
                flattened = np.concatenate([flattened, np.array([v])])
        return flattened
    
    def _change_channel_quality(self):
        # State settings
        velocity = 100   # km/h
        snr_thr = 15
        snr_ave = snr_thr + 10
        f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
        speedoflight = 300000   # km/sec
        f_d = velocity/(3600*speedoflight)*f_0  # Hz
        packettime = 300    # us
        fdtp = f_d*packettime/1e6
        # 0: Good, 1: Bad
        TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
        TRAN_00 = 1 - TRAN_01
        # TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_11 = 1 - TRAN_10

        if self.channel_quality == 0:  # Bad state
            if self._stepfunc(TRAN_00, random.random()) == 0: # 0 to 0
                channel_quality = 0
            else:   # 0 to 1
                channel_quality = 1
        else:   # Good state
            if self._stepfunc(TRAN_11, random.random()) == 0: # 1 to 1
                channel_quality = 1
            else:   # 1 to 0
                channel_quality = 0
    
        return channel_quality
    
    def _is_buffer_empty(self):
        return self.leftbuffers == BUFFERSIZE
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # State reset
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.current_time = 0
        self.current_aois = np.zeros([NUMNODES], dtype=float)
        self.node_location = BUFFERSIZE * np.ones([NUMNODES], dtype=int)
        self.node_aoi = np.zeros([NUMNODES], dtype=float)
        
        self.leftslots = round(BEACONINTERVAL / TIMEEPOCH)
        self.leftbuffers = BUFFERSIZE
        self.current_time = 0
        self.consumed_energy = 0
        self.inbuffer_info_node = np.zeros([BUFFERSIZE], dtype=int)
        self.inbuffer_info_timestamp = np.zeros([BUFFERSIZE], dtype=int)
        self.insert_index = 0
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        return self.current_obs, self.info


    def probenqueue(self, dflog):
        self.current_time += TIMEEPOCH / BEACONINTERVAL
        self.current_aois += TIMEEPOCH / BEACONINTERVAL
        
        # Define condition that the elements of the dflog can enqueue.
        cond = ((dflog.time >= self.current_time*BEACONINTERVAL - TIMEEPOCH) 
                & (dflog.time < self.current_time*BEACONINTERVAL))
        
        # Extract target dflog
        targetdflog = dflog[cond][:self.leftbuffers]
        tnodenumber = min(len(targetdflog), self.leftbuffers)
        self.leftbuffers -= tnodenumber

        if tnodenumber == 0:
            pass
        else:
            enquenode = targetdflog.node.values.astype(int)
            enquenodetimestamp = targetdflog.timestamp.values.astype(int)

            self.inbuffer_info_node[self.insert_index:self.insert_index + tnodenumber] = enquenode
            self.inbuffer_info_timestamp[self.insert_index:self.insert_index + tnodenumber] = enquenodetimestamp
            self.insert_index += tnodenumber

            self.node_location, self.node_aoi = self._get_node_info(self.inbuffer_info_node, self.inbuffer_info_timestamp)
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

    def step(self, action):  # 여기 해야 함.
        reward = 0
        # 0: FORWARD
        if action == 0:
            if self._is_buffer_empty():
                pass
            else:
                dequenodes = self.inbuffer_info_node[self.inbuffer_info_node != 0][:FRAMEAGGLIMIT]
                dequenodeaoi_timestamps = self.inbuffer_info_timestamp[self.inbuffer_info_node != 0][:FRAMEAGGLIMIT]
                num_dequenodes = len(dequenodes)
                
                if self.channel_quality == 0:
                    for dequenode, dequenodeaoi_timestamp in zip(dequenodes, dequenodeaoi_timestamps):
                        self.current_aois[dequenode - 1] = self.current_time - (dequenodeaoi_timestamp/BEACONINTERVAL)
                
                # Left-shift bufferinfo
                self.inbuffer_info_node[:-num_dequenodes] = self.inbuffer_info_node[num_dequenodes:]
                self.inbuffer_info_node[-num_dequenodes:] = 0
                self.inbuffer_info_timestamp[:-num_dequenodes] = self.inbuffer_info_timestamp[num_dequenodes:]
                self.inbuffer_info_timestamp[-num_dequenodes:] = 0
                self.leftbuffers += num_dequenodes
                self.insert_index -= num_dequenodes
            reward -= POWERCOEFF*0.308
            self.consumed_energy += 280 * 1.1 * FRAMETIME    # milliamperes * voltage * time

        # 1: Flush
        elif action == 1:
            self.inbuffer_info_node.fill(0)
            self.inbuffer_info_timestamp.fill(0)
            self.leftbuffers = BUFFERSIZE
            self.insert_index = 0

        # 2: Leave
        elif action == 2:
            pass
        
        self.node_location, self.node_aoi = self._get_node_info(self.inbuffer_info_node, self.inbuffer_info_timestamp)
        self.channel_quality = self._change_channel_quality()
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        self.leftslots -= 1
        done = self.leftslots <= 0
        
        # if self.current_aois.max() >= (PEAKAOITHRES / BEACONINTERVAL):
        reward -= np.clip(self.current_aois - (PEAKAOITHRES / BEACONINTERVAL), 0, None).sum()
        # count the number of nodes whose aoi is less than PEAKAOITHRES / BEACONINTERVAL
        # reward += np.count_nonzero(self.current_aois < (PEAKAOITHRES / BEACONINTERVAL)) * (1/NUMNODES)
        
        return self.current_obs, reward, False, done, self.info

    def step_rlaqm(self, action, dflog):  # 여기 해야 함.
        reward = 0
        # 0: FORWARD
        if action == 0:
            if self._is_buffer_empty():
                pass
            else:
                dequenode = self.inbuffer_info_node[0]
                dequenodeaoi_timestamp = self.inbuffer_info_timestamp[0]
                
                if self.channel_quality == 0:
                    self.current_aois[dequenode - 1] = self.current_time - (dequenodeaoi_timestamp/BEACONINTERVAL)
                
                # Left-shift bufferinfo
                self.inbuffer_info_node[:-1] = self.inbuffer_info_node[1:]
                self.inbuffer_info_node[-1] = 0
                self.inbuffer_info_timestamp[:-1] = self.inbuffer_info_timestamp[1:]
                self.inbuffer_info_timestamp[-1] = 0
                self.leftbuffers += 1
                self.insert_index -= 1
            reward -= 0.308
            self.consumed_energy += 280 * 1.1 * FRAMETIME    # milliamperes * voltage * time

        # 1: DISCARD
        elif action == 1:
            if self._is_buffer_empty():
                pass
            else:
                # Left-shift bufferinfo
                self.inbuffer_info_node[:-1] = self.inbuffer_info_node[1:]
                self.inbuffer_info_node[-1] = 0
                self.inbuffer_info_timestamp[:-1] = self.inbuffer_info_timestamp[1:]
                self.inbuffer_info_timestamp[-1] = 0
                self.leftbuffers += 1
                self.insert_index -= 1
            reward -= 0.154
            self.consumed_energy += 50 * 1.1 * FRAMETIME    # milliamperes * voltage * time

        # 2: SKIP
        elif action == 2:
            pass
        
        self.node_location, self.node_aoi = self._get_node_info(self.inbuffer_info_node, self.inbuffer_info_timestamp)
        self.channel_quality = self._change_channel_quality()
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        self.leftslots -= 1
        done = self.leftslots <= 0
        
        # Calculate link utilization
        link_utilization = (len(dflog[dflog.time/BEACONINTERVAL < self.current_time]) * FRAMETIME) / (self.current_time * TIMEEPOCH)

        # reward = (link_utilization**2 - 0.5) + (2/(1+(self.current_aois[self.current_aois != np.inf].mean()*BEACONINTERVAL/1000)/5) - 1.5)

        return self.current_obs, reward, False, done, self.info
    
        

    
    def render(self):
        # Implement viz
        pass

    # def getblockcount(self):
    #     """
    #     :return: scalar
    #     """
    #     return self.blockcount

    def getaoi(self):
        """
        :return: scalar
        """
        return self.aoi

    
