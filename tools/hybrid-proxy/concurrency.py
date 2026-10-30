"""FIFO session ownership. The condition mutex is never held during a turn."""
from collections import deque
import threading


class SessionQueue:
    def __init__(self):
        self.condition = threading.Condition()
        self.waiters = deque()

    def reserve(self, skip_busy=False):
        with self.condition:
            if skip_busy and self.waiters:
                return None
            ticket = Ticket(self)
            self.waiters.append(ticket)
            self.condition.notify_all()
            return ticket


class Ticket:
    def __init__(self, queue):
        self.queue = queue

    def __enter__(self):
        try:
            with self.queue.condition:
                self.queue.condition.wait_for(lambda: self.queue.waiters[0] is self)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        with self.queue.condition:
            self.queue.waiters.remove(self)
            self.queue.condition.notify_all()
