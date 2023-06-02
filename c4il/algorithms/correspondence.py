import enum
import itertools
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
from gym.spaces.utils import flatten
from sklearn import neighbors, preprocessing
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.rewards import reward_wrapper
from imitation.util import logger as imit_logger
from imitation.util import util



