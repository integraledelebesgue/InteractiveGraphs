import ctypes


class Tracker:
    def __init__(self, value):
        self.__value = value
        self.history = []

    def update(self):
        self.history.append(self.__value[0])


def foo():
    x = ctypes.c_int(0)
    tracker = Tracker(ctypes.pointer(x))

    for _ in range(10):
        tracker.update()
        x.value += 1

    return tracker


print(foo().history)
