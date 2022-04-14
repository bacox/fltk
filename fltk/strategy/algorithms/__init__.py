from .Deadline import Deadline
from .Offloading import Offloading
from .TiFL import TiFL
from .Vanilla import Vanilla
from fltk.util.definitions import Algorithm


def get_algorithm(name: Algorithm):
    enum_type = Algorithm(name.value)
    algorithm_dict = {
        Algorithm.vanilla: Vanilla,
        Algorithm.tifl: TiFL,
        Algorithm.deadline: Deadline,
        Algorithm.offloading: Offloading
    }
    return algorithm_dict[enum_type]
