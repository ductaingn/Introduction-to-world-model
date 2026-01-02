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

    v_optim = train_vision_model(
        agent=agent,
        n_epochs=20,
        batch_size=256,
        learning_rate=1e-4,
        device=device
    )

    r_optim = train_reasoning_model(
        agent=agent,
        n_epochs=8,
        batch_size=16,
        rollout_time_length=agent.reasoning_model.rollout_time_length,
        learning_rate=1e-4,
        device=device
    )

    # train_policy_model(
    #     agent=agent,
    #     n_episodes=2,
    #     batch_size=64
    # )

    if save_path is not None:
        agent.save_checkpoint(save_path, vision_optimizer=v_optim, reasoning_optimizer=r_optim)


if __name__ == "__main__":
    train_pipeline(
        device="cuda:0",
        save_path="checkpoint/trained_world_model.pt"
    )