import hydra
import torch
from c4il.utils.factory import make_ppo_models #, make_test_env 
# from torchrl.envs import ExplorationType, set_exploration_type
from c4il.utils.files import get_configs_path
from c4il.utils.factory import make_collector
import tqdm

@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="base")
def collect_trajectories(cfg):

    # Create actor model as per your training script
    actor, _, _ = make_ppo_models(cfg.experiment)

    # Load saved weights
    save_path = "/home/alper/c4il/data/weights/weights.pt"
    actor.load_state_dict(torch.load(save_path))
    actor.eval()

    # Create collector (for training environment)
    collector, _ = make_collector(cfg.experiment, policy=actor)

    # Initialize trajectory collection
    trajectories = []
    
    # Progress bar to track collection
    pbar = tqdm.tqdm(total=cfg.experiment.collector.total_frames)
    collected_frames = 0

    # Collect trajectories using the collector
    for data in collector:
        # Process and save the collected trajectory
        trajectories.append(data.cpu())  

        # Update progress
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Break condition to limit the number of collected trajectories (optional)
        if collected_frames >= cfg.experiment.collector.total_frames:
            break

    # Save collected trajectories
    torch.save(trajectories, "/home/alper/c4il/data/demos/trajectories2.pt")

    print(f"Collected {len(trajectories)} trajectories and saved to file.")

if __name__ == "__main__":
    collect_trajectories()
