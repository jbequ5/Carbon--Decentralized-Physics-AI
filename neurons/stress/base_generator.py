# neurons/stress/base_generator.py

"""
Base classes and registry for stress generation.

Follows docs/STRESS_TEST_DESIGN.md
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type

from .stress_models import StressTestSet, StressVariant


class BaseStressGenerator(ABC):
    """
    Abstract base class for all stress generators.
    """

    @abstractmethod
    def generate(
        self,
        challenge_id: str,
        physics_class: str,
        seed: int,
        difficulty: float = 0.5
    ) -> List[StressVariant]:
        """
        Generate a list of StressVariant objects.
        """
        pass


class StressGeneratorRegistry:
    """
    Registry that maps physics classes to appropriate stress generators.
    """

    def __init__(self):
        self._generators: Dict[str, List[BaseStressGenerator]] = {}

    def register(self, physics_class: str, generator: BaseStressGenerator):
        if physics_class not in self._generators:
            self._generators[physics_class] = []
        self._generators[physics_class].append(generator)

    def get_generators(self, physics_class: str) -> List[BaseStressGenerator]:
        return self._generators.get(physics_class, [])


# Global registry instance
stress_registry = StressGeneratorRegistry()
