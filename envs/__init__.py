from .point_mass import PointMassEnv
from .nonholonomic_car import NonholonomicCarEnv
from .two_link_arm import TwoLinkArmEnv
from .planar_pusher import PlanarPusherEnv


def make_envs():
    return [
        PointMassEnv(),
        NonholonomicCarEnv(),
        TwoLinkArmEnv(),
        PlanarPusherEnv(),
    ]


def get_env(name: str):
    for env in make_envs():
        if env.name == name:
            return env
    raise KeyError(name)
