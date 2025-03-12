## MocapDM:
Class for mocap data
mocap = MocapDM()
mocap.load_mocap(path)
mocap.data_config[frame][joint ind] -- qpos(35,)
mocap.data_vel[frame] --  qvel(34,)

## MyEnv: 
Customized gym env supporting qpos+qvel as observation space(Box(35+34,)), action space Box((28,))

## test.xml:
humanoid xml
