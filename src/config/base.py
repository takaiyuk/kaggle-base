from abc import ABC
from dataclasses import dataclass


@dataclass
class Basic:
    pass


@dataclass
class Column:
    pass


@dataclass
class Feature:
    pass


@dataclass
class Kfold:
    pass


@dataclass
class Model:
    pass


@dataclass
class Params:
    pass


@dataclass
class BaseFeConfig(ABC):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    kfold: Kfold = Kfold()


@dataclass
class BaseRunConfig(ABC):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    kfold: Kfold = Kfold()
    model: Model = Model()
    params: Params = Params()
