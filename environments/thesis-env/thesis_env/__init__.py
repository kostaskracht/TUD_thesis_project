from gym.envs.registration import register

register(
    id='thesis-env-v1',
    entry_point='thesis_env.envs:ThesisEnv'
)

register(
    id='cartpole-env-v1',
    entry_point='thesis_env.envs:CartPoleEnvi'
)