import torch
import numpy as np
from mushroom_rl.environments import Gym
# from train_invert_pendulum import Network
from train_cart_pole import LinearModel
# from mushroom_rl.policy import EpsGreedy

from mushroom_rl.policy import GaussianTorchPolicy
# from mushroom_rl.approximators import Regressor


def collect_rollout(policy_weights_path, rollout_size, env_id, rollout_params, horizon, gamma, policy_params):
    
    # Load policy weights
    #loaded_dict = torch.load(policy_weights_path)
    #weight_tensor = loaded_dict['linear.weight']
    #bias_tensor = loaded_dict['linear.bias']
    #weights = np.concatenate((weight_tensor.numpy().flatten(), bias_tensor.numpy().flatten()))
    weights = torch.load(policy_weights_path)
    # Create environment
    env = Gym(env_id, horizon, gamma)

     # Create LinearModel
    #linear_model = LinearModel(env.info.observation_space.shape, env.info.action_space.shape)

    # Set the loaded weights
    #linear_model.load_state_dict(loaded_dict)

    # Create policy
    #policy = EpsGreedy(linear_model, **policy_params)
    policy = GaussianTorchPolicy(LinearModel, env.info.observation_space.shape, env.info.action_space.shape, **policy_params)
    policy.set_weights(weights)
    
    # Create rollout dictionary
    rollout_dict = {
        'env_id': env_id,
        'rollout_size': rollout_size,
        'rollout_params': rollout_params,
        'trajectories': []
    }
    
    # Collect trajectories
    for i in range(rollout_size):
   
        # Reset environment
        obs = env.reset()
        done = False

        # gaussian noise for random initial position
        obs += np.random.normal(0, 1, size= (4,))

        # Collect trajectory
        states = []
        actions = []

        # Initialize variables
        upright_steps_limit = 100
        upright_steps = 0
        threshold = 0.05
        goal_obs = np.array([1, 0, 0])
        i = 0 # Initialize the counter for the while loop

        while not done:
            # Get action from policy
            action = policy.draw_action(obs)
    
            # Take step in environment
            next_obs, _, done, _ = env.step(action)

            # Render
            env.render()

            # Save state and action
            states.append(obs.tolist())
            actions.append(action.tolist())

            # Check if pendulum is in the upright position
            if np.isclose(obs[0], goal_obs[0], threshold).all():
                upright_steps += 1
            else:
                upright_steps = 0

            # Finish the episode if pendulum is in the upright position for more than 10 steps
            if upright_steps >= upright_steps_limit:
                done = True

            # Update observation
            obs = next_obs

            # Increment the counter
            i += 1

            # Break out of the loop if too many iterations have passed
            if i >= 800:
                break

            # Print information
            print(f"Iteration: {i}")
            print(f"Current observation: {obs}")
            print(f"Goal observation: {goal_obs}")
            print(f"Consecutive upright steps: {upright_steps}")
            print(f"Done: {done}\n")

        # Save trajectory to rollout dictionary
        rollout_dict['trajectories'].append({'states': states, 'actions': actions})
        
    # Save rollout dictionary
    torch.save(rollout_dict, '/home/alper/c4il/data/demos/demos_cartpole.pt')

if __name__ == '__main__':

    # Define variables
    policy_weights_path = '/home/alper/c4il/data/weights/weights_cartpole.pt'
    rollout_size = 10
    env_id = 'CartPole-v0'
    rollout_params = {'rollout_type': 'expert'}
    horizon=50 
    gamma=.99
    policy_params = dict(
        #std_0=1.,
        #n_features=32,
        #epsilon=0.,  # Set epsilon to 0 for deterministic actions during rollout
        #use_cuda=torch.cuda.is_available()

    )

    collect_rollout(policy_weights_path, rollout_size, env_id, rollout_params, horizon, gamma, policy_params=policy_params)