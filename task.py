from dataclasses import dataclass
from threading import Thread
from queue import Queue


@dataclass
class ProgressUpdate:
    progress: float
    message: str


class Task:
    thread: Thread
    output: Queue

    def __init__(self, target, prompt):
        def thread_run(msg, out):
            target(msg, out)
            out.put(ProgressUpdate(1, 'task complete'))

        self.output = Queue()
        self.thread = Thread(target=thread_run, args=(prompt, self.output))
        self.thread.start()

    def join(self):
        self.thread.join()

    def read_progress(self) -> ProgressUpdate:
        return self.output.get()
