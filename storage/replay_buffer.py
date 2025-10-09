import csv
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, filepath: str, capacity: int = 100000, timeframe: str = None):
        """
        :param filepath: CSV 파일 경로
        :param capacity: 버퍼 최대 크기
        :param timeframe: '15m', '1d' 중 필터링할 타임프레임 (None이면 전체 로드)
        """
        self.filepath = filepath
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.timeframe = timeframe
        self._load_from_csv()

    def _load_from_csv(self):
        """CSV에서 데이터 로드"""
        try:
            with open(self.filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.timeframe and row["timeframe"] != self.timeframe:
                        continue
                    self.buffer.append(self._convert_row(row))
        except FileNotFoundError:
            pass  # 파일 없으면 새로 시작

    def _convert_row(self, row: dict):
        """CSV에서 읽은 문자열 dict → 숫자형 dict 변환"""
        return {
            "timestamp": int(row["timestamp"]),
            "datetime": row["datetime"],
            "market": row["market"],
            "timeframe": row["timeframe"],
            "open": float(row["open"]) if row["open"] else None,
            "high": float(row["high"]) if row["high"] else None,
            "low": float(row["low"]) if row["low"] else None,
            "close": float(row["close"]) if row["close"] else None,
            "volume": float(row["volume"]) if row["volume"] else None,
            "kimchi_premium": float(row["kimchi_premium"]) if row["kimchi_premium"] else None,
        }

    def add(self, row: dict):
        """메모리에 추가 (이미 숫자형 dict 가정)"""
        self.buffer.append(row)

    def sample(self, batch_size: int):
        """랜덤 샘플"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self):
        return len(self.buffer)


if __name__ == "__main__":
    # 15분봉만 불러오기
    buffer_15m = ReplayBuffer("storage/trade_data.csv", timeframe="15m")
    print("15m buffer size:", buffer_15m.size())
    batch = buffer_15m.sample(5)
    for b in batch:
        print(b)

    # 1일봉만 불러오기
    buffer_1d = ReplayBuffer("storage/trade_data.csv", timeframe="1d")
    print("1d buffer size:", buffer_1d.size())
    print(buffer_1d.sample(3))