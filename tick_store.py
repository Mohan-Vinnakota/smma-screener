from collections import deque
import time
from practice import Tick

class TickStore:
    def __init__(self, max_minutes=120):
        self.max_seconds = max_minutes * 60
        self._ticks = deque()

    def add(self, tick):
        self._ticks.append(tick)
        cutoff = time.time() - self.max_seconds
        while self._ticks and self._ticks[0].timestamp < cutoff:
            self._ticks.popleft()

    def etq(self, minutes):
        cutoff = time.time() - minutes * 60
        return sum(t.ltq for t in self._ticks if t.timestamp >= cutoff)

    def count(self):
        return len(self._ticks)

    def avg_ltq(self, minutes):
        cutoff = time.time() - minutes * 60
        recent = [t.ltq for t in self._ticks if t.timestamp >= cutoff]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def avg_ltp(self, minutes):
        cutoff = time.time() - minutes * 60
        recent = [t for t in self._ticks if t.timestamp >= cutoff]
        if not recent:
            return None
        total_qty = sum(t.ltq for t in recent)
        if total_qty == 0:
            return sum(t.ltp for t in recent) / len(recent)
        return sum(t.ltp * t.ltq for t in recent) / total_qty

if __name__ == "__main__":
    store = TickStore()
    store.add(Tick("IRFC", 195.5, 800, 195.4, 1200000, 195.6, 1500000))
    store.add(Tick("IRFC", 195.6, 1200, 195.5, 1100000, 195.7, 1400000))
    store.add(Tick("IRFC", 195.7, 500, 195.6, 1300000, 195.8, 1600000))
    print(f"Total ticks stored: {store.count()}")
    print(f"ETQ last 5 min: {store.etq(5)}")
    print(f"ETQ last 20 min: {store.etq(20)}")
    print(f"Avg LTQ last 2 min: {store.avg_ltq(2)}")
    print(f"Avg LTQ last 5 min: {store.avg_ltq(5)}")