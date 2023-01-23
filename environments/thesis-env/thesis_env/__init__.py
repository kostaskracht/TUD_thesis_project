from gym.envs.registration import register

register(
    id='thesis-env-v1',
    entry_point='thesis_env.envs:RoadEnv'
)