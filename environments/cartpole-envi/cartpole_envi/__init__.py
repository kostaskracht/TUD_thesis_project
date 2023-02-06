from gym.envs.registration import register

register(
    id='cartpole-envi-v1',
    entry_point='cartpole_envi.envs:CartPoleEnvi'
)