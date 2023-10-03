import hydra
import torch
from c4il.utils.factory import make_test_env, make_ppo_models
from torchrl.envs import ExplorationType, set_exploration_type
from c4il.utils.files import get_configs_path
from c4il.utils.factory import make_collector

@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="base")
def collect_trajectories(cfg):

    # Create actor model as per your training script
    actor, _, _ = make_ppo_models(cfg.experiment)

    # Load saved weights
    save_path = "/home/alper/c4il/data/weights/weights.pt"
    actor.load_state_dict(torch.load(save_path))
    actor.eval()

    _, state_dict = make_collector(cfg.experiment, policy=actor)
    # Create test environment
    test_env = make_test_env(cfg.experiment.task, state_dict)

    # Initialize trajectory collection
    trajectories = []

    # Collect trajectories
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        # Generate a complete episode
        td_test = test_env.rollout(
            policy=actor,
            max_steps=10_000_000,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
        ).clone()

        # Process and save the collected trajectory
        trajectories.append(td_test)

    # Save collected trajectories
    torch.save(trajectories, "/home/alper/c4il/data/demos/trajectories.pt")


if __name__ == "__main__":
    collect_trajectories()
