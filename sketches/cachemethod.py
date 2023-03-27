class Simple:
    def __init__(self):
        self.a = None
        self.__pv = None

    def __foo(self):
        print("Foo")


smpl = Simple()

print(hasattr(smpl, '__pv'))
