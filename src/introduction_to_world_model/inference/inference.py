import gymnasium as gym
import miniworld

if __name__ == "__main__":
    env = gym.make("MiniWorld-Sign-v0")
    obs, _ = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"observation shape: {next_obs['obs'].shape}, {next_obs['goal']}\naction: {action}\nreward: {reward}"
        )
