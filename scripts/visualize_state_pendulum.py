import matplotlib.pyplot as plt
import torch
from torchrl.envs.libs.gym import GymEnv

# NOTE: this does not work

device = "cpu" if not torch.cuda.is_available() else "cuda:0"

env = GymEnv("Pendulum-v1", from_pixels=True, pixels_only=False, device=device)
tensordict = env.reset()

# Set the state to the swing up state
swing_up_state = torch.tensor([1., 0., 0], device=device)        
tensordict.set('observation', swing_up_state)
tensordict = env.reset(tensordict)
# tensordict.set('action', torch.tensor([0.0], device=device))  # set dummy action
# tensordict = env.step(tensordict)


# visualize environment
plt.imshow(tensordict.get('pixels').squeeze().cpu().numpy())
plt.show()
