import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
from mocap.mocap import MocapDM
import random
import mujoco
from PIL import Image
import os
from scipy.spatial.transform import Rotation as R
import pyquaternion
from mocap.mocap_util import BODY_JOINTS_IN_DP_ORDER,DOF_DEF
import math

MOCAP_PATH = "./walk.txt"
XML_PATH = "./test.xml"
RENDER_FOLDER = "./mujoco_render"

class MyEnv(MujocoEnv):
    """TODO: observation space init"""
    def __init__(self, xml_path):
        # self.width = 960 # size of renderer
        # self.height = 640 # size of renderer
        self.metadata["render_modes"] = [
            "human",
            "rgb_array",
            "depth_array",

        ]
        super().__init__(XML_PATH,frame_skip = 5,observation_space = None)

        # self.obs_dim = self.data.qpos.shape[0] + self.data.qvel.shape[0] + self.data.cinert.size

        # for simplicity only use qpos and qvel for now
        
        self.obs_dim = self.data.qpos.shape[0] + self.data.qvel.shape[0]
        # print(self.obs_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        self.render_mode = "rgb_array"
        self.render_path = RENDER_FOLDER
        self.curr_frame_ind = 0

        '''
        self.mocap.data_config[frame][joint ind] -- qpos
        self.mocap.data_vel[frame][joint ind] -- qvel
        '''
        self.mocap = MocapDM()
        self.load_mocap(MOCAP_PATH)
        # print("mocap data shape: ", self.mocap.data_config[0].shape)

        # random initialization of ref state
        self.reference_state_init()
        self.idx_curr = -1
        self.idx_tmp_count = -1
        self.step_len=1 # what's this

        self.curr_rollout_step = 0
        self.max_step = 500

    def set_renderer(self):
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            width = self.width,
            height = self.height,
            camera_id=self.camera_id,
            camera_name=self.camera_name
        )
    """Try to load mocap data from mocap path"""
    def load_mocap(self,mocap_path):
        self.mocap.load_mocap(mocap_path)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)
    
    """TODO: Change this if want to include more dim to obs space"""
    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)

    """TODO: Change this if want to include more dim in observation"""
    def _get_obs(self):

        # cinert = self.data.cinert.flatten().copy()
        position = self.data.qpos.flatten().copy()
        velocity = self.data.qvel.flatten().copy()
        # position = self.data.qpos.flatten().copy()[7:] # ignore root joint
        # velocity = self.data.qvel.flatten().copy()[6:] # ignore root joint
        return np.concatenate((position, velocity))
    
    """Random initialization"""
    def reference_state_init(self):
        self.curr_rollout_step=0
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        # self.idx_init = 0
        self.idx_curr = self.idx_init
        self.idx_tmp_count = 0
        # print("ref state initialization succeeded.")

    """TODO: reward functions."""
    def step(self, action):
        # self.step_len = int(self.mocap_dt // self.model.opt.timestep)
        self.step_len = 1
        # step_times = int(self.mocap_dt // self.model.opt.timestep)
        step_times = 1
        # pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, step_times)
        # pos_after = mass_center(self.model, self.sim)

        observation = self._get_obs()
        
        truncated = self.curr_rollout_step>self.max_step

        # reward_alive = 1.0
        reward_alive = 0.
        pos_reward = 0.
        vel_reward = 0.
        '''
        reward_obs = self.calc_config_reward()
        reward_acs = np.square(data.ctrl).sum()
        reward_forward = 0.25*(pos_after - pos_before)

        reward = reward_obs - 0.1 * reward_acs + reward_forward + reward_alive

        info = dict(reward_obs=reward_obs, reward_acs=reward_acs, reward_forward=reward_forward)
        '''
        # reward = self.calc_config_reward()
        # TODO: modify reward_alive
        pos_reward = self.calc_pos_reward()
        vel_reward = self.calc_vel_reward()
        if self.curr_rollout_step!=0 and self.curr_rollout_step%3==0:
            # print("curr simul step:" ,self.curr_rollout_step)

            self.idx_curr += 1 # ind of curr_frame
            self.idx_curr = self.idx_curr % self.mocap_data_len
        self.curr_rollout_step+=1

        # reward = reward_alive
        info = dict()
        done = self.is_done() 
        if done:
            reward_alive = -1 # magic number
        
        done = done | truncated
        reward = 0.75*pos_reward+0.25*vel_reward+reward_alive
        # print("curr_reward:", reward)
        
        # self.save_render_image(self.mujoco_renderer.render(self.render_mode),self.curr_frame_ind)
        # self.curr_frame_ind+=1
        # self.mujoco_renderer.close()

        # with mujoco.Renderer(self.model) as renderer:
        #     mujoco.mj_forward(self.model,self.data)
        #     renderer.update_scene(self.data)
        #     pixels = renderer.render()
        #     self.save_render_image(pixels,self.curr_frame_ind)
        #     self.curr_frame_ind+=1

        return observation, reward, done, truncated, info
    
    def calc_pos_reward(self, interpolate = True):
        assert len(self.mocap.data) != 0


        pos_diff = 0
        curr_pos_offset = 7
        for curr_joint in BODY_JOINTS_IN_DP_ORDER:
            dof = DOF_DEF[curr_joint]
            if "knee" in curr_joint or "hip" in curr_joint or "ankle" in curr_joint:
                scalar = 3
            else:
                scalar = 1
            if dof==1:
                curr_pos_diff = self.calc_pos_errs_interpolation(dof,curr_pos_offset)
                pos_diff += curr_pos_diff*scalar


            if dof==3:

                curr_pos_diff = self.calc_pos_errs_interpolation(dof,curr_pos_offset)
                # print(curr_pos_diff,pos_diff)
                pos_diff += curr_pos_diff*scalar

            curr_pos_offset+=dof

        pos_reward = math.exp(-2*pos_diff)

        # self.idx_curr += 1 # ind of curr_frame
        # self.idx_curr = self.idx_curr % self.mocap_data_len

        return pos_reward

    def calc_pos_errs_interpolation(self, dof, curr_offset):

        def slerp(q1, q2, t):
            q1 = R.from_quat(q1)
            q2 = R.from_quat(q2)
            return R.slerp(t, [q1, q2]).as_quat()
        last_frame_ind = (self.curr_rollout_step // 3)%self.mocap_data_len
        next_frame_ind = (last_frame_ind+1)%self.mocap_data_len
        t = (self.curr_rollout_step%3)/3
        if dof == 1:
            diff = (1-t)*self.mocap.data_config[last_frame_ind][curr_offset:curr_offset+dof]+t*self.mocap.data_config[next_frame_ind][curr_offset:curr_offset+dof]
            return diff[0]**2
        if dof==3:
            last_target = self.mocap.data_config[last_frame_ind][curr_offset:curr_offset+dof]
            quat1 = self.euler2quat(last_target[0],last_target[1],last_target[2],scalar_first=True)
            next_target = self.mocap.data_config[last_frame_ind][curr_offset:curr_offset+dof]
            quat2 = self.euler2quat(next_target[0],next_target[1],next_target[2],scalar_first=True)

            q_inpl = slerp(quat1,quat2,t)
            
            curr_pos = self.data.qpos[curr_offset:curr_offset+dof]
            q_curr = self.euler2quat(curr_pos[0],curr_pos[1],curr_pos[2],True)
            q_inpl = pyquaternion.Quaternion(q_inpl[3],-q_inpl[0],-q_inpl[1],-q_inpl[2]).normalised
            q_curr = pyquaternion.Quaternion(q_curr[3],-q_curr[0],-q_curr[1],-q_curr[2]).normalised
            return (q_inpl*q_curr).angle**2


    
    # def calc_pos_errs(self,curr_config, target_config): # suppose for each single joint
    #     # assuming both are w x y z
    #     if len(curr_config) == 3: # DOF 3
    #         curr_quat = self.euler2quat(curr_config[0],curr_config[1],curr_config[2],True)
    #         tar_quat = self.euler2quat(target_config[0],target_config[1],target_config[2],True)
            
    #         curr_quat_conjugate = pyquaternion.Quaternion(curr_quat[0],-curr_quat[1],-curr_quat[2],-curr_quat[3]).normalised
    #         target_quat = pyquaternion.Quaternion(tar_quat[0],tar_quat[1],tar_quat[2],tar_quat[3]).normalised
    #         diff = target_quat*curr_quat_conjugate

    #         return diff.angle**2
    #     else: # DOF 1
    #         # print(curr_config,target_config)
    #         diff = (curr_config-target_config)[0]**2
    #         # print("returning pos diff: ", diff)
    #         return diff
    

    def calc_vel_reward(self):
        assert len(self.mocap.data) != 0

        vel_diff = 0
        curr_vel_offset = 6
        for curr_joint in BODY_JOINTS_IN_DP_ORDER:
            dof = DOF_DEF[curr_joint]
            if dof==1:
                target_config = self.mocap.data_vel[self.idx_curr][curr_vel_offset:curr_vel_offset+dof]
                curr_config = self.data.qvel[curr_vel_offset:curr_vel_offset+dof]
                curr_vel_diff = self.calc_pos_errs(curr_config,target_config)
                vel_diff += curr_vel_diff

            if dof==3:
                # dof = 4
                target_config = self.mocap.data_vel[self.idx_curr][curr_vel_offset:curr_vel_offset+dof]
                curr_config = self.data.qvel[curr_vel_offset:curr_vel_offset+dof]
                curr_vel_diff = self.calc_pos_errs(curr_config,target_config)
                vel_diff += curr_vel_diff

            curr_vel_offset+=dof

        vel_reward = math.exp(-.1*vel_diff)

        # self.idx_curr += 1 # ind of curr_frame
        # self.idx_curr = self.idx_curr % self.mocap_data_len

        return vel_reward

    def calc_vel_errs(self,curr_vel, target_vel):
        # both in quat
        return sum((curr_vel-target_vel)**2)

    def euler2quat(self,x,y,z,scalar_first = True): # ignore the scalar_first argument
        r = R.from_euler('xyz', [x, y, z], degrees=False)
    
        # Convert to quaternion
        quaternion = r.as_quat()  # returns [x, y, z, w] (vector part first) # need to check if this is consistent with 
        return quaternion
            
    '''
    alive reward: Check z position of weighted CoM across all body elements.
    TODO: Other source?
    '''
    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        # xpos = self.sim.data.xipos
        xpos = self.data.xipos #CoM of all bodies in global coordinate
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
        done = bool((z_com < 0.7) or (z_com > 2.0))
        return done
    
    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        # qvel = self.init_qvel
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.idx_tmp_count = -self.step_len
        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()
    
    def save_render_image(self,image,ind):
        outpath = os.path.join(RENDER_FOLDER,f"{ind}.png")
        im = Image.fromarray(image)
        im.save(outpath)

    
if __name__=="__main__":
    testEnv = MyEnv("")
    assert testEnv is not None, "No env created"
    print("Successfully created env.")
    
