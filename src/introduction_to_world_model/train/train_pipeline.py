from introduction_to_world_model.env.env import Env
from introduction_to_world_model.model.world_model import WorldModel
from introduction_to_world_model.train.train_vision_model import train_vision_model
from introduction_to_world_model.train.train_reasoning_model import train_reasoning_model
from introduction_to_world_model.train.train_policy_model import train_policy_model


def train_pipeline(    
    device: str = "cpu",
    save_path: str | None = None,
):
    env = Env()

    agent = WorldModel(env.observation_space, env.action_space)

    print("Collecting rollout data")
    agent.collect_data(env, n_step=10000)

    train_vision_model(
        agent=agent,
        n_episodes=2,
        batch_size=64,
    )

    train_reasoning_model(
        agent=agent,
        n_episodes=2,
        batch_size=64,
        rollout_time_length=8
    )

    train_policy_model(
        agent=agent,
        n_episodes=2,
        batch_size=64
    )

    agent.save_checkpoint(save_path)


if __name__ == "__main__":
    train_pipeline(...)