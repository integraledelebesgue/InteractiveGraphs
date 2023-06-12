import threading
import time


class MyClass(threading.Thread):
    def __init__(self):
        super().__init__()
        self.__running = False
        self.__paused = False
        self.__resume = threading.Condition()

    def run(self) -> None:
        self.__running = True

        while self.__running:
            with self.__resume:
                if self.__paused:
                    self.__resume.wait()

            print("Thread...")
            time.sleep(0.5)

    def pause(self):
        self.__paused = True

    def resume(self):
        self.__paused = False
        with self.__resume:
            self.__resume.notify()

    def stop(self):
        self.__running = False


thread = MyClass()
thread.start()

time.sleep(2)

thread.pause()

time.sleep(3)

thread.resume()

time.sleep(2)

thread.stop()

