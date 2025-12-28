from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import Final


@dataclass(frozen=True)
class Constants:
    eps: float
    mu: float

    @cached_property
    def c(self):
        return 1 / sqrt(self.eps * self.mu)


FREE_SPACE_CONSTS: Final[Constants] = Constants(
    eps=8.854_187_818_8e-12, mu=1.256_637_061_271e-6
)
