import csv
import os
from datetime import datetime

class CSVWriter:
    def __init__(self, filepath: str, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        # 파일 없으면 헤더 추가
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def write(self, row: dict):
        """데이터 한 줄 기록"""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# 사용 예시
if __name__ == "__main__":
    writer = CSVWriter("storage/trade_data.csv",
                       ["timestamp", "price", "volume", "kimchi_premium", "position", "reward", "done"])
    writer.write({
        "timestamp": datetime.utcnow().isoformat(),
        "price": 65000,
        "volume": 1.2,
        "kimchi_premium": 3.1,
        "position": "LONG",
        "reward": 0.05,
        "done": False
    })