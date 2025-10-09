from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", "state action reward next_state done aux")

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buf = deque(maxlen=maxlen)

    def append(self, t: Transition):
        self.buf.append(t)

    def __len__(self):
        return len(self.buf)

    def sample(self, n):
        n = min(n, len(self.buf))
        return random.sample(self.buf, n)

    def clear(self):
        self.buf.clear()