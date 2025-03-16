<!-- ## MocapDM:
Class for mocap data

mocap = MocapDM()

mocap.load_mocap(path)

mocap.data_config[frame][joint ind] -- qpos(35,)

mocap.data_vel[frame] --  qvel(34,)

## MyEnv: 
Customized gym env supporting qpos+qvel as observation space(Box(35+34,)), action space Box((28,))

## test.xml:
humanoid xml

## vis:
    env = MyEnv() --change render mode to "human"

    loop over data:

      env.set_state(qpos,qvel)

      mujuco.mj_forward(env.model,env.data)

      env.render()

      ind+=1 -->

## Pytorch Implementation of DeepMimic

