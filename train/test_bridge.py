from wdb_env import WdbGymEnv
env = WdbGymEnv(stage=2)
obs, _ = env.reset()
print("Initial obs shape:", obs.shape)
for _ in range(10):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        break
env.close()
print("OK - real bonk-core WDB bridge works!")
