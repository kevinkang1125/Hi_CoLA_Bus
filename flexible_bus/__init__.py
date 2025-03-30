from gym.envs.registration import register

register(
    id='FlexibleBus-v0', 
    entry_point='flexible_bus.envs:FlexibleBusEnv'
)