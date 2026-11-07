"""
PhysicsGuard - EV DC Fast-Charging Physics Simulator & Digital Twin
===================================================================
A beginner-friendly physical simulation package modeling an EV battery pack
(89s32p NMC cells) and liquid cold-plate cooling loop during DC fast charging.
"""

from . import params
from .plant import Plant
from .twin import Twin
from .controller import ChargerBMS
from .simulate import run_session
from .scenarios import Scenario, build_scenario
from . import faults
from . import attacks
from .guard import PhysicsGuard, Verdict

__all__ = [
    "params",
    "Plant",
    "Twin",
    "ChargerBMS",
    "run_session",
    "Scenario",
    "build_scenario",
    "faults",
    "attacks",
    "PhysicsGuard",
    "Verdict",
]
