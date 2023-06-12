from enum import Enum


class MyClass(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


lst = [
    (MyClass.FIRST, [1, 2, 3]),
    (MyClass.SECOND, (0, 2)),
    (MyClass.THIRD, 2137)
]

