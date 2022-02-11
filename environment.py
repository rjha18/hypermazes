import time
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt;
import networkx as nx;

	
LEFT	= 0
RIGHT	= 1
UP	= 2
DOWN	= 3

actions_dict = {0:'LEFT',1:'RIGHT',2:'UP',3:'DOWN'};

valid_actions = set((LEFT,RIGHT,UP,DOWN));

UNIT = 40 #pixels per square



class init_error(Exception):
	message = '';

class grid_missing_values(init_error):
	message = 'Error: Grid is missing values (has to be a rectangle).';
	
class grid_invalid_value(init_error):
	message = 'Error: Grid values have to be in {0,1,2,3}.';

class no_starting_state(init_error):
	message = 'Error: No starting state provided.';

class no_goal_state(init_error):
	message = 'Error: No goal state provided.';
	
class invalid_action(init_error):
	message = 'Error: Action must be in '+str(valid_actions)+'.';

class start_is_goal(init_error):
	message = 'Starting state and goal state must be different.';


class gridworld_env(tk.Tk):

	def __init__(self,fnm=None,step_penalty=-0.01,goal_reward=1.0,display=True,gamma=0.99):
		try:
			fp = open(fnm, 'r')
			fp.close()
		except OSError:
			print("Map file cannot be opened.")
			raise OSError()
			
		map_data = np.array(pd.read_csv(fnm,header=None,delimiter=' '));
		gridworld_env(map_data, step_penalty, goal_reward, display, gamma)

	def __init__(self,map_data,step_penalty=-0.01,goal_reward=1.0,display=True,gamma=0.99):
		self.map_data = map_data

		if np.isnan(self.map_data).any():
			self.handle_exception(grid_missing_values());

		self.H,self.W = self.map_data.shape;
		self.step_penalty = step_penalty;
		self.goal_reward = goal_reward;
		self.display = display;
		self.gamma = gamma;
		
		
		self.state_lookup = {};
		self.state_counts = {};
		
		self.states = np.zeros((0,2));
		self.blocked = np.zeros((0,2));
		
		self.valid_values = set((0,1));
		
		self.policy_init = False;
		self.task_init = False;
		
		self.starting_state = -1
		self.curr_state = -1
		self.goal_state = -1
		
		self.start_reset_counter = 0;
		
		
		#### self.model has one entry per valid state and action
		#### The entry is a tuple (next_state, reward, goal_reached)
		
		self.model = {};
		
		
		for h in range(self.H):
			for w in range(self.W):
				cell = self.map_data[h,w];
				state = np.array([h,w]);

				if cell not in self.valid_values:
					self.handle_exception(grid_invalid_value())
	
				if cell==0:
					self.states = np.concatenate([self.states,state.reshape([1,2])],axis=0);
					self.state_lookup[self.state_to_key(state)] = self.states.shape[0]-1;
					self.state_counts[self.state_to_key(state)] = 0;
					
					self.model[self.state_to_key(state)] = {a : {'next_state':''} for a in range(4)}
				
					left_state = state;
					right_state = state;
					up_state = state;
					down_state = state;
					
		
					if w-1>=0:
						if self.map_data[h,w-1]==0:
							left_state = np.array([h,w-1]);
							
					if w+1<=self.W-1:
						if self.map_data[h,w+1]==0:
							right_state = np.array([h,w+1]);
								
					if h-1>=0:
						if self.map_data[h-1,w]==0:
							up_state = np.array([h-1,w]);
								
					if h+1<=self.H-1:
						if self.map_data[h+1,w]==0:
							down_state = np.array([h+1,w]);
							
		
					self.model[self.state_to_key(state)] = {LEFT:left_state,RIGHT:right_state,UP:up_state,DOWN:down_state};
						
				else:
					self.blocked = np.concatenate([self.blocked,state.reshape([1,2])],axis=0);

		self.num_states = self.states.shape[0];
		self.num_actions = len(valid_actions);
		self.state_sz = 2;
		
		self.A = np.zeros((self.num_states,self.num_states));

		# print("symmetric adj matrix")
		for state in self.states:
				state_key = self.state_to_key(state);
				state_idx = self.state_lookup[state_key];
				neighbours = self.model[state_key]
				
				for action in range(4):
					neighbour_idx = self.state_lookup[self.state_to_key(neighbours[action])];

					if self.A[state_idx,neighbour_idx]==0:
						self.A[state_idx,neighbour_idx] = 1;
						
						self.A[neighbour_idx,state_idx] = 1;
						
		self.graph = nx.from_numpy_matrix(self.A)
		if nx.number_connected_components(self.graph) > 1:
			raise OSError("Disconnected Graph!")

		self.graph_metric = np.zeros(self.A.shape);
		paths = dict(nx.all_pairs_shortest_path(self.graph))
		for i in range(self.num_states):
			for j in range(i+1,self.num_states):
				# dist = len(nx.shortest_path(self.graph,source=i,target=j))-1;
				dist = len(paths[i][j]) - 1
				
				self.graph_metric[i,j] = dist;
				self.graph_metric[j,i] = dist;
		
		
		if self.display:
			super(gridworld_env, self).__init__()
			self.title('Gridworld Environment')
			self.geometry('{0}x{1}'.format(self.H*UNIT,self.W*UNIT))
			self.canvas = self._build_canvas()

	
	def _build_canvas(self):
	
		canvas = tk.Canvas(self, bg='white',height=self.H*UNIT,width=self.W*UNIT)

		for c in range(0,self.W*UNIT,UNIT):
			x0, y0, x1, y1 = c, 0, c, self.H*UNIT
			canvas.create_line(x0, y0, x1, y1)
		for r in range(0,self.H*UNIT,UNIT):
			x0, y0, x1, y1 = 0, r, self.H*UNIT, r
			canvas.create_line(x0, y0, x1, y1)

		for block in self.blocked:
			canvas.create_rectangle(block[1]*UNIT,block[0]*UNIT,(block[1]+1)*UNIT,(block[0]+1)*UNIT,fill="gray")

		canvas.pack()

		return canvas
			
	
	def handle_exception(self,exception):
		print(exception.message)
		raise exception

	def state_to_key(self,state):
		state = state.reshape([-1]);
		return str(np.int32(state[0]))+'_'+str(np.int32(state[1]));
		
	def lookup_state(self,state):
		return self.state_lookup[self.state_to_key(state)];
			
		
	def get_stats_xy(self):
		states = self.states;
		
		min_x = np.min(states[:,0]);	
		min_y = np.min(states[:,1]);	
		max_x = np.max(states[:,0]);	
		max_y = np.max(states[:,1]);	
		
		rx = (max_x+min_x)/2;
		ry = (max_y+min_y)/2;
		
		sx = max_x-min_x;
		sy = max_y-min_y;
		
		return rx,ry,sx,sy;
		
		
		
		
		
	def lookup_distance_by_state(self,s,t):
		s_idx = self.lookup_state(s)
		t_idx = self.lookup_state(t)
		return self.graph_metric[s_idx,t_idx];
	
	def q_value_by_state(self,s,t):
		D = self.lookup_distance_by_state(s,t);
		return self.q_value(D);
	
	def q_value_by_index(self,s_idx,t_idx):
		D = self.graph_metric[s_idx,t_idx];
		return self.q_value(D);
	
	def q_value(self,distance):
		gain = np.power(self.gamma,distance-1);
		return (gain*self.goal_reward - self.step_penalty*(1-gain)/(1-self.gamma)).reshape([-1,1]);
	
			
	def get_neighbours(self,state):
	
		state_entry = self.model[self.state_to_key(state)];
		neighbours = np.zeros((self.num_actions,2));
		
		for action in range(self.num_actions):
			neighbours[action] = state_entry[action];
					
		return neighbours;

	def pad_with(self, vector, pad_width, iaxis, kwargs):
		pad_value = kwargs.get('padder', 10)
		vector[:pad_width[0]] = pad_value
		vector[-pad_width[1]:] = pad_value

	def q_angle_by_state(self, heatmap, state):
		h = state[0] + 1
		w = state[1] + 1

		states = [heatmap[x, y] for x in [h-1, h, h+1] for y in [w-1, w, w+1]]
		angles = [3, 2, 1, 4, 0, 5, 6, 7]
		return angles[np.argmax(states[:4]+states[5:])]
		
	def plot_vals(self,fnm=None,random=False):
	
		if random:
			idx = np.random.randint(0,self.num_states)
			self.goal_state = self.states[idx]
	
		heatmap = np.zeros(self.map_data.shape);
		counter = 0;
		
		goal_state = self.get_goal_state();
		for h in range(self.H):
			for w in range(self.W):
				cell = self.map_data[h,w];

				if cell==0:
					heatmap[h,w] = self.q_value_by_state(self.states[counter],self.get_goal_state());
					counter += 1;
				elif cell==1:
					heatmap[h,w] = -1.0;

		plt.imshow(heatmap)
		plt.colorbar()
		if self.display:
			plt.show()
		if fnm:
			plt.savefig(fnm)
		
	def generate_Q(self, verbosity=0):
		Q = []
		for i in range(len(self.states)):
			goal_state = self.states[i]
			heatmap = np.zeros(self.map_data.shape);
			counter = 0;
			for h in range(self.H):
				for w in range(self.W):
					cell = self.map_data[h,w];

					if cell==0:
						heatmap[h,w] = self.q_value_by_state(self.states[counter],goal_state);
						counter += 1;
					elif cell==1:
						heatmap[h,w] = -1.0;
			directions = [];
			heatmap = np.pad(heatmap, 1, self.pad_with, padder=-1.0)
			for j in range(len(self.states)):
				state = self.states[j].astype(np.int)	
				cell = self.map_data[state[0], state[1]];	
				if cell==0:
					directions.append(self.q_angle_by_state(heatmap, state));

			Q.append(directions)
			if verbosity != 0 and i % verbosity == 0:
				print(f"Finished round {i}")
		return np.array(Q)
	
	def plot_results(self, results, results_raw, goal, fnm=None, random=False, target_num=8, wall_num=-1):
		if random:
			idx = np.random.randint(0,self.num_states)
			self.goal_state = self.states[idx]
	
		heatmap = np.zeros(self.map_data.shape);
		angle = np.zeros((*self.map_data.shape, 2));
		counter = 0;

		for h in range(self.H):
			for w in range(self.W):
				cell = self.map_data[h,w];

				if cell==0:
					if counter == goal:
						heatmap[h,w] = target_num
						angle[h,w] = 0
					else:
						heatmap[h,w] = results[counter]
						angle[h,w] = results_raw[counter]
					counter += 1;
				elif cell==1:
					heatmap[h,w] = wall_num;
					angle[h,w] = 0

		
		plt.clf()
		plt.imshow(heatmap, alpha=0.7)
		plt.colorbar()

		U = angle[:,:,1]
		V = angle[:,:,0]
		
		U = U / ((U**2 + V**2) ** 0.5)
		V = V / ((U**2 + V**2) ** 0.5)

		plt.quiver(U,V,pivot="middle")
		if self.display:
			plt.show()
		if fnm:
			plt.savefig(fnm)

	def plot_Q(self, Q, Q_raw, goal, fnm=None, random=False, target_num=8, wall_num=-1):
		Q = np.expand_dims(Q[goal], 1)
		self.plot_results(Q, Q_raw, goal, fnm, random, target_num, wall_num)

	def split_pairs(self,split):

		N = self.num_states;
		pairs_num = N*N-N;
		
		
		pairs = np.zeros((pairs_num,2));
		pairs_vals = np.zeros((pairs_num,1));
		
		counter = 0;
		
		for s_idx in range(N):
			for t_idx in range(N):
				if s_idx!=t_idx:
					pairs[counter] = np.array([s_idx,t_idx]).reshape([1,2]);
					pairs_vals[counter] = self.q_value_by_index(s_idx,t_idx);
					counter += 1;
		

		train_num = np.int32(np.floor(split*pairs_num));
		test_num = pairs_num - train_num;

		self.train_set = np.random.choice(pairs_num,size=train_num,replace=False);
		self.test_set = np.setdiff1d(np.array(range(pairs_num)),self.train_set);
		
		self.train_num = train_num;
		self.test_num = test_num;
		self.pairs = np.int32(pairs);
		self.pairs_vals = pairs_vals;
		
		if counter!=pairs_num:
			print("# of pairs does not match!")
			input()
		if len(np.intersect1d(self.train_set,self.test_set))!=0:
			print("# of pairs does not match!")
			input()
		if len(np.union1d(self.train_set,self.test_set))!=pairs_num:
			print("# of pairs does not match!")
			input()
		
		
	def set_random_goal_pair(self,mode="Train"):
	
		if self.task_init and self.display:
			self.canvas.delete(self.start_square);
			self.canvas.delete(self.goal_square);
			
		
		if mode=="Train":
			p_idx = np.random.choice(self.train_set);
		else:	
			p_idx = np.random.choice(self.test_set);
			
		pair = self.pairs[p_idx];
		
		self.starting_state = self.states[pair[0]];
		self.goal_state = self.states[pair[1]];
		
		if self.display:
			start_x = self.starting_state[0];
			start_y = self.starting_state[1];
			self.start_square = self.canvas.create_rectangle(start_y*UNIT,start_x*UNIT,(start_y+1)*UNIT,(start_x+1)*UNIT,fill="green")
			
			goal_x = self.goal_state[0];
			goal_y = self.goal_state[1];
			self.goal_square = self.canvas.create_rectangle(goal_y*UNIT,goal_x*UNIT,(goal_y+1)*UNIT,(goal_x+1)*UNIT,fill="yellow")
			
		self.reset_episode();
		self.task_init = True;
		
		
	def render(self):
		self.canvas.update()
			
	def reset_episode(self):
		
		self.curr_state = self.starting_state;
		
		if self.display:
			if self.task_init:
				self.canvas.delete(self.agent_square);
		
			cur_x = self.curr_state[0];
			cur_y = self.curr_state[1];
			self.agent_square = self.canvas.create_rectangle(cur_y*UNIT,cur_x*UNIT,(cur_y+1)*UNIT,(cur_x+1)*UNIT,fill="blue")
			
		self.render()
		
	def set_path_length(self,length):
		self.path_length = length;
		
	def set_goal_state(self,t):
		self.goal_state = t;
		self.render();
		
	def get_curr_state(self):
		return self.curr_state.reshape([1,2]);
			
	def get_goal_state(self):
		return self.goal_state.reshape([1,2]);
		
	def get_starting_state(self):
		return self.starting_state.reshape([1,2]);
		
	def get_all_states(self):
		return self.states;
		
	def states_equal(s1,s2):
		if s1[0]==s2[0] and s1[1]==s2[1]:
			return True;
		return False;
		
	def render_policy(self,policy):
		
		if self.display:
			if self.policy_init:
				for arrow in self.policy_arrows:
					self.canvas.delete(arrow)
					
			policy_arrows = [];


			for sidx in range(self.num_states):
			
				state = self.states[sidx];
				best_action = np.int32(policy[sidx]);

				center_x = state[0]+0.5;
				center_y = state[1]+0.5;
				
				end_x = center_x;
				end_y = center_y;
					
				
				if best_action==LEFT:
					end_y -= 0.5;
				elif best_action==RIGHT:
					end_y += 0.5;
				elif best_action==UP:
					end_x -= 0.5;
				elif best_action==DOWN:
					end_x += 0.5;
				else:
					# write exception
					pass;
				
				policy_arrow = self.canvas.create_line(center_y*UNIT,center_x*UNIT,end_y*UNIT,end_x*UNIT,arrow=tk.LAST);
				self.canvas.tag_raise(policy_arrow)
					
				policy_arrows += [policy_arrow];
				
			
			self.policy_arrows = policy_arrows;
			self.policy = policy;
				
			self.policy_init = True;
			self.render()
		
	
	def take_action(self,action,render=True):

		if action not in valid_actions:
			self.handle_exception(invalid_action());
		
		prev_state = self.curr_state;

		self.curr_state = self.model[self.state_to_key(prev_state)][action];
		self.state_counts[self.state_to_key(self.curr_state)] += 1;
		
		goal_reached = np.int32(np.array_equal(self.curr_state,self.goal_state));
		
		reward = self.goal_reward*goal_reached + self.step_penalty*(1-np.square(goal_reached));
		
		one_hot_action = np.zeros((1,self.num_actions));
		one_hot_action[0,action] = 1.0;

		action_outcome = {'curr_state':prev_state,'next_state':self.curr_state,\
			'curr_action':one_hot_action,'curr_reward':reward,'goal_reached':goal_reached,'goal_state':self.goal_state};
		
		
		if render:
			if self.display:
				diff = (self.curr_state-prev_state)*UNIT;
				self.canvas.move(self.agent_square,diff[1],diff[0])
				self.render()
		
		return action_outcome;
		
		
		
	def split_states(self,split):

		train_num = np.int32(np.floor(split*self.num_states));
		test_num = self.num_states - train_num;
		
		self.train_set = np.random.choice(self.num_states,size=train_num,replace=False);
		self.test_set = np.setdiff1d(np.array(range(self.num_states)),self.train_set);
		
		self.train_num = train_num;
		self.test_num = test_num;
		
		
	def get_nn(self):
	
		length = self.path_length;

		s = self.starting_state;
				
		s_idx = self.state_lookup[self.state_to_key(s)];
		candidate_vertices = self.graph_metric[s_idx,:];
		nn_idx = np.where((candidate_vertices<=length) & (candidate_vertices>0))[0];

		nn = self.states[nn_idx];

		nn_vals = np.zeros((0,1));

		for t in nn:
			q_val = self.q_value_by_state(s,t);
			nn_vals = np.concatenate([nn_vals,q_val.reshape([1,1])],axis=0);

		return nn,nn_vals;	

	def set_random_goal_nn(self,mode="Train"):
	
		if self.task_init and self.display:
			self.canvas.delete(self.start_square);
			self.canvas.delete(self.goal_square);
		
		if mode=="Train":
			s_idx = np.random.choice(self.train_set);
		else:	
			s_idx = np.random.choice(self.test_set);
			
		
		self.starting_state = self.states[s_idx];
		
		nn,nn_vals = self.get_nn();
	
		nn_size = nn.shape[0];
		self.goal_state = nn[np.random.choice(nn_size)];
		
		if self.display:
			start_x = self.starting_state[0];
			start_y = self.starting_state[1];
			self.start_square = self.canvas.create_rectangle(start_y*UNIT,start_x*UNIT,(start_y+1)*UNIT,(start_x+1)*UNIT,fill="green")
			
			goal_x = self.goal_state[0];
			goal_y = self.goal_state[1];
			self.goal_square = self.canvas.create_rectangle(goal_y*UNIT,goal_x*UNIT,(goal_y+1)*UNIT,(goal_x+1)*UNIT,fill="yellow")
			
		self.reset_episode();
		self.task_init = True;
		
		return nn,nn_vals;
		
	def set_internal(self,s_idx,t_idx):
	
		if self.task_init and self.display:
			self.canvas.delete(self.start_square);
			self.canvas.delete(self.goal_square);
		
		self.starting_state = self.states[s_idx];
		self.goal_state = self.states[t_idx];
			
		if self.display:
			start_x = self.starting_state[0];
			start_y = self.starting_state[1];
			self.start_square = self.canvas.create_rectangle(start_y*UNIT,start_x*UNIT,(start_y+1)*UNIT,(start_x+1)*UNIT,fill="green")
			
			goal_x = self.goal_state[0];
			goal_y = self.goal_state[1];
			self.goal_square = self.canvas.create_rectangle(goal_y*UNIT,goal_x*UNIT,(goal_y+1)*UNIT,(goal_x+1)*UNIT,fill="yellow")
			
		self.reset_episode();
		self.task_init = True;		    
