from config import SMMA_FAST, SMMA_SLOW

class SMMACalculator:
    def __init__(self, period):
        self.period = period
        self.value = None
        self._buffer = []
        self._ready = False

    def update(self, price):
        if not self._ready:
            self._buffer.append(price)
            if len(self._buffer) == self.period:
                self.value = sum(self._buffer) / self.period
                self._ready = True
        else:
            self.value = (self.value * (self.period - 1) + price) / self.period
        return self.value


class CrossoverDetector:
    def __init__(self):
        self.fast = SMMACalculator(SMMA_FAST)
        self.slow = SMMACalculator(SMMA_SLOW)
        self._prev_fast = None
        self._prev_slow = None

    def update(self, price):
        f = self.fast.update(price)
        s = self.slow.update(price)

        signal = None
        if f and s and self._prev_fast and self._prev_slow:
            if self._prev_fast <= self._prev_slow and f > s:
                signal = "BUY"
            elif self._prev_fast >= self._prev_slow and f < s:
                signal = "SELL"

        self._prev_fast = f
        self._prev_slow = s
        return signal


if __name__ == "__main__":
    detector = CrossoverDetector()

    prices = (
        [100] * 130 +
        [100 + i * 0.2 for i in range(100)] +
        [120] * 50 +
        [120 - i * 0.3 for i in range(100)]
    )

    for i, price in enumerate(prices):
        signal = detector.update(price)
        if signal:
            print(f"Tick {i}: {signal} signal at price {round(price, 2)}")