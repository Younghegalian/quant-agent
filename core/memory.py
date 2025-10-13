from collections import deque, namedtuple

Transition = namedtuple("Transition", "state action reward done")

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buf = deque(maxlen=maxlen)

    def append(self, t: Transition):
        self.buf.append(t)

    def __len__(self):
        return len(self.buf)

    def sample(self, n):
        n = min(n, len(self.buf))
        buf_list = list(self.buf)
        return buf_list[-n:]

    def clear(self):
        self.buf.clear()