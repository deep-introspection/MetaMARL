from abc import ABC, abstractmethod
from typing import Optional, Callable
from core.optimizers.graph import OptimizerGraph
from core.optimizers.config import OptimizerConfig
from core.signal.base import Signal
from logging import Logger

# why are we inheriting from ABC ?
# Should we inherit from gym.Algorithm ?

class Optimizer(ABC):
    """
    Docstring for Optimizer
    """

    # data owned by the optimizer
    config : OptimizerConfig

    signal : Optional[Signal] #immutable # O, really unsure about this. shouldnt there jsut be a policy ?

    graph : Optional[OptimizerGraph]

    metrics : Optional[MetricsLogger] # necessary ?

    offline_data : Optional[OfflineData] # necessary ?

    logger_creator : Optional[Callable[[], Logger]]


    # this is the default configuration as soon as an optimizer is created

    def __init__(self):
        # TODO
        # where we initialize exactly what we can modify in the optimizer api that we expose
        pass

    def __str__(self) -> str :
        # TODO serialize class
        pass


    # what is a class method decorator doing really
    # what is cls vs self
    @classmethod
    def from_config(cls, config: OptimizerConfig) -> Optimizer :
        """
        Docstring for from_config
        
        :param cls: Description
        :param config: Description
        :type config: OptimizerConfig
        """
        # equivalent to setup in ray.algorithm
        return cls(config=config)
    
    @classmethod
    def get_default_config(cls) -> OptimizerConfig :
        """
        Docstring for get_default_config
        
        :param cls: Description
        :return: Description
        :rtype: OptimizerConfig
        """
        return cls.config
    
    @classmethod
    def from_checkpoint(cls, file_path: Any) -> Optimizer :
        """
        Docstring for from_checkpoint
        
        :param cls: Description
        :param file_path: Description
        :type file_path: Any
        :return: Description
        :rtype: Optimizer
        """
        # TODO
        pass
    
    # Accessors
    # def __getattribute__(self, name):
    #     return super().__getattribute__(name)
    
    def get_signal(self) -> Signal :
        return self.signal
    
    
    def get_graph(self) -> OptimizerGraph :
        """
        Docstring for get_graph
        
        :param self: Description
        :return: Description
        :rtype: OptimizerGraph
        """
        return self.graph
    
    # Mutators
    # def __setattr__(self, name, value) -> Any:
    #     return super().__setattr__(name, value)

    def set_downstream(self) -> None :
        #TODO
        pass

    def set_upstream(self) -> None :
        #TODO
        pass

    def set_graph(self, graph: OptimizerGraph) -> None:
        """
        Docstring for remove_node
        
        :param self: Description
        :param node_id: Description
        :type node_id: NodeID
        """
        self.graph = graph
        return None


    @abstractmethod
    def step(self, singal : Optional[Signal]) -> ResultDict :
        """
        Docstring for run
        
        :param self: Description
        """
        # this would be multiple steps in MDP algorithms
        # not sure about the resultDict here
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self) -> ResultDict :
        """
        Docstring for evaluate
        
        :param self: Description
        :return: Description
        :rtype: Any
        """
        raise NotImplementedError
    
    def save_checkpoint(self) -> None :
        # TODO
        pass

    

    

    
    

    



