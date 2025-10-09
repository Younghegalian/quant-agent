import aiohttp, asyncio, sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from typing import List, Tuple, Optional
import pytz
from loguru import logger

from model import Candle
from config import Config
from storage.db_writer import CSVWriter

UTC = timezone.utc
KST = pytz.timezone("Asia/Seoul")


class UpbitDataPipeline:
    def __init__(self):
        self.base_url = "https://api.upbit.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self._init_database()

        # 2) ✅ CSVWriter 초기화 (하나의 파일에 timeframe 컬럼으로 구분)
        #    Config.CSV_PATH가 없다면 추가해둬: 예) storage/trade_data.csv
        self.csv_writer = CSVWriter(
            filepath=getattr(Config, "CSV_PATH", "storage/trade_data.csv"),
            fieldnames=[
                "timestamp", "datetime", "market", "timeframe",
                "open", "high", "low", "close", "volume",
                "kimchi_premium"
            ]
        )

    # ----------------------------------------
    # DB 초기화 (기존 스키마 유지)
    # ----------------------------------------
    def _init_database(self):
        cur = self.db_conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS candles_15m (
                timestamp INTEGER PRIMARY KEY,
                datetime TEXT,
                market TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS candles_1d (
                timestamp INTEGER PRIMARY KEY,
                datetime TEXT,
                market TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                kimchi_premium REAL
            )
        """)
        self.db_conn.commit()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=Config.SSL_CONTEXT)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # ----------------------------------------
    # Upbit 캔들 수집
    # ----------------------------------------
    async def fetch_candles(self, market="KRW-USDT", interval=15, count=200, to=None) -> List[Candle]:
        if interval == 15:
            endpoint = f"{self.base_url}/candles/minutes/15"
        elif interval == 1440:
            endpoint = f"{self.base_url}/candles/days"
        else:
            raise ValueError("interval must be 15 or 1440")

        params = {"market": market, "count": min(count, 200)}
        if to:
            params["to"] = to.strftime("%Y-%m-%dT%H:%M:%S")

        async with self.session.get(endpoint, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Upbit fetch fail ({market}): {resp.status} {text[:150]}")
                return []
            data = await resp.json()
            candles = []
            for item in data:
                ts = datetime.fromisoformat(item["candle_date_time_utc"]).replace(tzinfo=UTC)
                candles.append(
                    Candle(
                        timestamp=ts,
                        open=float(item["opening_price"]),
                        high=float(item["high_price"]),
                        low=float(item["low_price"]),
                        close=float(item["trade_price"]),
                        volume=float(item["candle_acc_trade_volume"])
                    )
                )
            return candles

    # ----------------------------------------
    # Binance BTCUSDT 1일봉 (메모리용)
    # ----------------------------------------
    async def _fetch_binance_btcusdt(self) -> float:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        async with self.session.get(url) as resp:
            data = await resp.json()
            return float(data["price"])

    # ----------------------------------------
    # 김치프리미엄 계산 (UPDATE만)
    # ----------------------------------------
    async def _update_kimchi_premium(self):
        """
        ✅ 환율(USD/KRW) 기반 김치프리미엄 계산
        candles_1d.kimchi_premium = (Upbit KRW-USDT / 환율 - 1) * 100
        - 환율은 오전 9시 고시환율 (무료 API)
        - 기존 데이터는 절대 수정하지 않음
        - 트랜잭션으로 일괄 UPDATE
        """
        cur = self.db_conn.cursor()

        # 환율 데이터 (무료 API)
        url = "https://api.manana.kr/exchange/rate/USD/KRW.json"
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"환율 API 오류: {resp.status} {text[:150]}")
                    return
                fx_data = await resp.json()
                rate = 1 / float(fx_data[0]["rate"])
        except Exception as e:
            logger.error(f"환율 데이터 수집 실패: {e}")
            return

        # Upbit KRW-USDT 일봉 가져오기
        cur.execute("""
            SELECT timestamp, close FROM candles_1d
            WHERE market='KRW-USDT'
            ORDER BY timestamp ASC
        """)
        rows = cur.fetchall()
        if not rows:
            logger.warning("KRW-USDT 일봉 없음 → 김프 업데이트 스킵")
            return

        # 김치프리미엄 계산
        updates = []
        for ts, close in rows:
            premium = round((float(close) / rate - 1.0) * 100.0, 3)
            updates.append((premium, ts))

        # 트랜잭션으로 일괄 UPDATE
        cur.execute("BEGIN IMMEDIATE")
        cur.executemany("""
            UPDATE candles_1d
               SET kimchi_premium = ?
             WHERE market='KRW-USDT' AND timestamp = ?
        """, updates)
        self.db_conn.commit()

        logger.info(f"✅ 환율 기반 김치프리미엄 업데이트 완료 ({len(updates)} rows)")

        # 3-b) ✅ 1d CSV 기록: 김프가 반영된 “완성 레코드”만 append
        #     (중복 방지를 위해 1d는 _save_candle 에서 기록하지 않음)
        for premium, ts in updates:
            cur.execute("""
                SELECT datetime, open, high, low, close, volume
                  FROM candles_1d
                 WHERE market='KRW-USDT' AND timestamp=?
            """, (ts,))
            row = cur.fetchone()
            if not row:
                continue
            dt, o, h, l, c, v = row
            self.csv_writer.write({
                "timestamp": ts,
                "datetime": dt,
                "market": "KRW-USDT",
                "timeframe": "1d",
                "open": o, "high": h, "low": l, "close": c, "volume": v,
                "kimchi_premium": premium
            })

        # 최근 5개 로그
        cur.execute("""
            SELECT datetime, close, kimchi_premium
            FROM candles_1d
            WHERE market='KRW-USDT'
            ORDER BY timestamp DESC LIMIT 5
        """)
        for dt, close, prem in cur.fetchall():
            logger.info(f"[김프] {dt} | USDT₩={close:.2f} | 환율={rate:.2f} | 김치={prem:.3f}%")

    # ----------------------------------------
    # 히스토리컬 데이터 수집
    # ----------------------------------------
    async def fetch_historical_data(self, market="KRW-USDT", days_back=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_15m, all_1d = [], []
        end_15m = datetime.now(UTC)
        end_1d = datetime.now(UTC)

        # 15분봉
        total_15m = days_back * 24 * 4
        for _ in range((total_15m // 200) + 1):
            batch = await self.fetch_candles("KRW-USDT", 15, 200, to=end_15m)
            if not batch:
                break
            for c in batch:
                self._save_candle(c, "KRW-USDT", "15m")
            all_15m.extend(batch)
            end_15m = batch[-1].timestamp - timedelta(minutes=15)
            await asyncio.sleep(0.05)

        # 1일봉
        for _ in range((days_back // 200) + 1):
            batch = await self.fetch_candles("KRW-USDT", 1440, 200, to=end_1d)
            if not batch:
                break
            for c in batch:
                self._save_candle(c, "KRW-USDT", "1d")
            all_1d.extend(batch)
            end_1d = batch[-1].timestamp - timedelta(days=1)
            await asyncio.sleep(0.05)

        # 김치프리미엄 UPDATE & 1d CSV 기록
        await self._update_kimchi_premium()

        df15 = pd.DataFrame([asdict(c) for c in all_15m])
        df1d = pd.DataFrame([asdict(c) for c in all_1d])
        return df15, df1d

    # ----------------------------------------
    # Save candle
    # ----------------------------------------
    def _save_candle(self, candle: Candle, market: str, timeframe: str):
        table = f"candles_{timeframe}"
        cur = self.db_conn.cursor()
        kst = candle.timestamp.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(f"""
            INSERT OR REPLACE INTO {table}
            (timestamp, datetime, market, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(candle.timestamp.timestamp()),
            kst,
            market,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        ))
        self.db_conn.commit()

        # 3-a) ✅ CSV 기록: 15m만 즉시 기록 (1d는 김프 반영 후 기록)
        if timeframe == "15m":
            self.csv_writer.write({
                "timestamp": int(candle.timestamp.timestamp()),
                "datetime": kst,
                "market": market,
                "timeframe": "15m",
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "kimchi_premium": None
            })