from dataclasses import dataclass, field
import time

@dataclass
class Tick:
    symbol: str
    ltp: float
    ltq: int
    bid_price: float
    bid_qty: int
    ask_price: float
    ask_qty: int
    timestamp: float = field(default_factory=time.time)

if __name__ == "__main__":
    t3 = Tick("IRFC", 195.5, 800, 195.4, 1200000, 195.6, 1500000)
    print(t3)
    print(f"{t3.symbol} arrived at timestamp: {t3.timestamp}")