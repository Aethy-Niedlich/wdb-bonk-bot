# train/test_bridge.py
from wdb_env import WdbGymEnv

env = WdbGymEnv(stage=1)
obs, _ = env.reset()
print("Initial obs shape:", obs.shape)
for i in range(10):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        break
env.close()
print("OK - WDB bridge and Box2D simulation work!")
