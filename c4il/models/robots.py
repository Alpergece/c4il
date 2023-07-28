from typing import Optional, List
from torch_robotics.torch_kinematics_tree.models.robot_tree import DifferentiableTree
from c4il.utils.files import get_urdf_path


class DifferentiableInvertedPendulum(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_urdf_path() / 'gym' / 'invertedpendulum.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_invertedpendulum"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableDoubleInvertedPendulum(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_urdf_path() / 'gym' / 'doubleinvertedpendulum.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_doubleinvertedpendulum"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableCartPole(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_urdf_path() / 'gym' / 'cartpole.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_cartpole"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)
