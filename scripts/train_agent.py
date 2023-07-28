import hydra
import omegaconf
from torchrl.record import VideoRecorder
from torchrl.envs import TransformedEnv
import torch
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import wandb

from c4il.utils.factory import (
    make_collector,
    make_data_buffer,
    make_logger,
    make_loss,
    make_optim,
    make_ppo_models,
    make_test_env,
)
from c4il.utils.files import get_configs_path


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="base")
def main(cfg):

    # Correct for frame_skip
    cfg.experiment.collector.total_frames = cfg.experiment.collector.total_frames // cfg.experiment.task.frame_skip
    cfg.experiment.collector.frames_per_batch = (
        cfg.experiment.collector.frames_per_batch // cfg.experiment.task.frame_skip
    )
    mini_batch_size = cfg.experiment.loss.mini_batch_size = (
        cfg.experiment.loss.mini_batch_size // cfg.experiment.task.frame_skip
    )

    model_device = cfg.experiment.optim.device
    actor, critic, critic_head = make_ppo_models(cfg.experiment)

    collector, state_dict = make_collector(cfg.experiment, policy=actor)
    data_buffer = make_data_buffer(cfg.experiment)
    loss_module, adv_module = make_loss(
        cfg.experiment.loss,
        actor_network=actor,
        value_network=critic,
        value_head=critic_head,
    )
    optim = make_optim(cfg.experiment.optim, loss_module)

    batch_size = cfg.experiment.collector.total_frames * cfg.experiment.task.num_envs
    num_mini_batches = batch_size // mini_batch_size
    total_network_updates = (
        (cfg.experiment.collector.total_frames // batch_size)
        * cfg.experiment.loss.ppo_epochs
        * num_mini_batches
    )

    scheduler = None
    if cfg.experiment.optim.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optim, total_iters=total_network_updates, start_factor=1.0, end_factor=0.1
        )

    logger = None
    if cfg.experiment.logger.backend:
        logger = make_logger(cfg.experiment.logger)
    test_env = make_test_env(cfg.experiment.task, state_dict)
    if cfg.experiment.logger.record_video:
        video_recorder = VideoRecorder(
            logger, tag='Eval Visualization'
        )
        test_env = TransformedEnv(test_env, video_recorder)

    record_interval = cfg.experiment.logger.log_interval
    pbar = tqdm.tqdm(total=cfg.experiment.collector.total_frames)
    collected_frames = 0

    # log hydra config
    logger.log_hparams(omegaconf.OmegaConf.to_container(
        cfg, resolve=True
    ))

    # Main loop
    r0 = None
    l0 = None
    frame_skip = cfg.experiment.task.frame_skip
    ppo_epochs = cfg.experiment.loss.ppo_epochs
    total_done = 0
    for data in collector:

        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Log end-of-episode accumulated rewards for training
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if logger is not None and len(episode_rewards) > 0:
            logger.log_scalar(
                "reward_training", episode_rewards.mean().item(), collected_frames
            )

        losses = TensorDict(
            {}, batch_size=[ppo_epochs, -(frames_in_batch // -mini_batch_size)]
        )
        for j in range(ppo_epochs):
            # Compute GAE
            with torch.no_grad():
                data = adv_module(data.to(model_device)).cpu()

            data_reshape = data.reshape(-1)
            # Update the data buffer
            data_buffer.extend(data_reshape)

            for i, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(model_device)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, i] = loss.detach()

                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )

                # Backward pass
                loss_sum.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=0.5
                )
                losses[j, i]["grad_norm"] = grad_norm

                optim.step()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad()

                # Logging
                if r0 is None:
                    r0 = data["next", "reward"].mean().item()
                if l0 is None:
                    l0 = loss_sum.item()
                pbar.set_description(
                    f"loss: {loss_sum.item(): 4.4f} (init: {l0: 4.4f}), reward: {data['next', 'reward'].mean(): 4.4f} (init={r0: 4.4f})"
                )
            if i + 1 != -(frames_in_batch // -mini_batch_size):
                print(
                    f"Should have had {- (frames_in_batch // -mini_batch_size)} iters but had {i}."
                )
        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        if logger is not None:
            for key, value in losses.items():
                logger.log_scalar(key, value.item(), collected_frames)
            logger.log_scalar("total_done", total_done, collected_frames)

        collector.update_policy_weights_()

        # Test current policy
        if (
            logger is not None
            and (collected_frames - frames_in_batch) // record_interval
            < collected_frames // record_interval
        ):

            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                test_env.eval()
                actor.eval()
                # Generate a complete episode
                td_test = test_env.rollout(
                    policy=actor,
                    max_steps=10_000_000,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                ).clone()
                test_env.transform.dump()
                logger.log_scalar(
                    "reward_testing",
                    td_test["next", "reward"].sum().item(),
                    collected_frames,
                )
                actor.train()
                del td_test


if __name__ == "__main__":
    main()