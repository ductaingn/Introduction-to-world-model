from enum import Enum


class ValidateMode(Enum):
    OOD = "Out-of-Distribution"
    ID = "In-Distribution"
