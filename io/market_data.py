import aiohttp, asyncio, pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
import ssl

BASE_UPBIT = "https://api.upbit.com/v1"
BASE_BINANCE = "https://api.binance.com/api/v3"


# ------------------------------------------------------------
# Upbit KRW-USDT 캔들 수집
# ------------------------------------------------------------
async def fetch_usdt_candles(interval="minutes/15", mode="agent", lookback=200,
                             tz="Asia/Seoul", save_csv=False, csv_name=None):
    """
    Upbit KRW-USDT 캔들 수집 (15분봉 / 일봉)
    mode:
      - 'agent': 최근 n개만
      - 'full': 가능한 과거까지
    """
    utc = timezone.utc
    kst = pytz.timezone(tz)
    all_candles = []

    async with aiohttp.ClientSession() as sess:
        if mode == "agent":
            # ✅ 최근 lookback개만 수집
            url = f"{BASE_UPBIT}/candles/{interval}"
            params = {"market": "KRW-USDT", "count": lookback}
            async with sess.get(url, params=params) as resp:
                data = await resp.json()

            if isinstance(data, dict) and "error" in data:
                print("⚠️ Upbit API error:", data["error"].get("message"))
                return pd.DataFrame()

            for d in sorted(data, key=lambda x: x["candle_date_time_utc"]):
                ts = datetime.fromisoformat(d["candle_date_time_utc"]).replace(tzinfo=utc)
                all_candles.append({
                    "timestamp_utc": ts,
                    "timestamp_kst": ts.astimezone(kst),
                    "open": float(d["opening_price"]),
                    "high": float(d["high_price"]),
                    "low": float(d["low_price"]),
                    "close": float(d["trade_price"]),
                    "volume": float(d["candle_acc_trade_volume"]),
                })

        elif mode == "full":
            # ✅ 가능한 한 오래된 데이터까지 전부 수집
            print(f"🚀 Start full data collection: KRW-USDT ({interval})")

            end = datetime.now(pytz.timezone("Asia/Seoul"))
            last_oldest = None
            request_count = 0

            while True:
                url = f"{BASE_UPBIT}/candles/{interval}"
                params = {
                    "market": "KRW-USDT",
                    "count": 200,
                    "to": end.strftime("%Y-%m-%d %H:%M:%S")
                }

                async with sess.get(url, params=params) as resp:
                    data = await resp.json()

                if isinstance(data, dict) and "error" in data:
                    print("⚠️ Upbit API error:", data["error"].get("message"))
                    break
                if not data:
                    print("✅ No more data, stopping.")
                    break

                # ✅ 정렬 및 누적
                for d in sorted(data, key=lambda x: x["candle_date_time_utc"]):
                    ts = datetime.fromisoformat(d["candle_date_time_utc"]).replace(tzinfo=utc)
                    all_candles.append({
                        "timestamp_utc": ts,
                        "timestamp_kst": ts.astimezone(kst),
                        "open": float(d["opening_price"]),
                        "high": float(d["high_price"]),
                        "low": float(d["low_price"]),
                        "close": float(d["trade_price"]),
                        "volume": float(d["candle_acc_trade_volume"]),
                    })

                if "minutes" in interval:
                    # 예: "minutes/15" → 15
                    step_min = int(interval.split("/")[1])
                    step = timedelta(minutes=step_min)
                else:
                    # "days"
                    step = timedelta(days=1)

                oldest = datetime.fromisoformat(data[-1]["candle_date_time_kst"])

                # ✅ 동일 timestamp 반복 방지
                if last_oldest and abs((last_oldest - oldest).total_seconds()) < 1:
                    print("⚠️ 동일 타임스탬프 반복 감지 → 루프 종료")
                    break
                last_oldest = oldest

                end = oldest - step
                request_count += 1

                if request_count % 10 == 0:
                    print(f"📈 {request_count}회 요청 완료, 총 {len(all_candles)}봉, oldest={oldest}")
                await asyncio.sleep(0.2)

    if not all_candles:
        print("⚠️ No data fetched.")
        return pd.DataFrame()

    # ✅ 정리
    df = pd.DataFrame(all_candles).drop_duplicates(subset="timestamp_utc")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if save_csv:
        safe_interval = interval.replace("/", "_")
        today = datetime.now().strftime("%Y%m%d")
        name = csv_name or f"USDT_15m_{today}.csv"
        df.to_csv(name, index=False)
        print(f"💾 CSV saved: {name} ({len(df)} rows)")

    return df

# ------------------------------------------------------------
# ✅ NBP 환율 수집 (기간별, 청크 분할 + 자동 보정)
# ------------------------------------------------------------
async def fetch_usdkrw_timeseries(start_date, end_date):
    """
    NBP API 기간 요청 제한(367일)을 피하기 위해 기간을 여러 청크로 나눠 요청.
    각 청크 실패 시 180→90→30→7→1일 단위로 축소 재시도.
    """
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    async def fetch_chunk(sess, s, e):
        url_usd = f"https://api.nbp.pl/api/exchangerates/rates/A/USD/{s}/{e}/?format=json"
        url_krw = f"https://api.nbp.pl/api/exchangerates/rates/A/KRW/{s}/{e}/?format=json"
        async with sess.get(url_usd) as r_usd, sess.get(url_krw) as r_krw:
            if r_usd.status != 200 or r_krw.status != 200:
                # ✅ 주말 구간이면 직전 평일 환율로 채우기
                print(f"⚠️ NBP 구간 실패 {s}~{e}: USD {r_usd.status}, KRW {r_krw.status}")
                return "WEEKEND_FILL"
            usd_data = await r_usd.json(content_type=None)
            krw_data = await r_krw.json(content_type=None)
        if "rates" not in usd_data or "rates" not in krw_data:
            return None
        usd_df = pd.DataFrame(usd_data["rates"])
        krw_df = pd.DataFrame(krw_data["rates"])
        if usd_df.empty or krw_df.empty:
            return None
        df = pd.merge(
            usd_df.rename(columns={"effectiveDate": "date", "mid": "usd_pln"}),
            krw_df.rename(columns={"effectiveDate": "date", "mid": "krw_pln"}),
            on="date", how="inner"
        )
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df["usdkrw"] = df["usd_pln"] / df["krw_pln"]
        if df["usdkrw"].median() > 5000:
            df["usdkrw"] /= 100.0
        return df[["date", "usdkrw"]]

    frames = []
    chunk_sizes = [360, 180, 90, 30, 7, 1]
    cur = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as sess:
        last_valid_df = None
        while cur <= end_dt:
            got = None
            for days in chunk_sizes:
                chunk_end = min(end_dt, cur + timedelta(days=days))
                s = cur.strftime("%Y-%m-%d")
                e = chunk_end.strftime("%Y-%m-%d")
                got = await fetch_chunk(sess, s, e)
                if got is not None:
                    break
            if isinstance(got, str) and got == "WEEKEND_FILL":
                # ✅ 주말 구간이면 직전 평일 환율로 채움
                weekend_dates = pd.date_range(start=cur, end=chunk_end)
                # 직전 평일 환율 찾기
                if last_valid_df is not None and not last_valid_df.empty:
                    last_rate = last_valid_df["usdkrw"].iloc[-1]
                else:
                    last_rate = 1300.0  # 기본값
                fill_df = pd.DataFrame({"date": weekend_dates, "usdkrw": [last_rate] * len(weekend_dates)})
                frames.append(fill_df)
                print(f"✅ 주말 구간 {cur}~{chunk_end}: 직전 평일 환율({last_rate})로 채움")
                cur = chunk_end + timedelta(days=1)
                continue
            if got is None:
                print(f"⚠️ {cur}~ 구간 완전 실패, 하루 건너뜀")
                cur += timedelta(days=1)
                continue
            frames.append(got)
            last_valid_df = got
            cur = got["date"].max().date() + timedelta(days=1)
            await asyncio.sleep(0.05)

    if not frames:
        print("⚠️ NBP 환율 수집 실패")
        return pd.DataFrame(columns=["date", "usdkrw"])

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    # ✅ 결측값 보정: 앞뒤 값으로 채움
    full["usdkrw"] = full["usdkrw"].ffill().bfill()
    return full[["date", "usdkrw"]]


# ------------------------------------------------------------
# ✅ BTC 기준 김치프리미엄 (USDT 기준 전체 동기화)
# ------------------------------------------------------------
async def fetch_kimchi_daily(mode="agent", lookback=200, save_csv=False):
    utc = timezone.utc
    kst = pytz.timezone("Asia/Seoul")

    async with aiohttp.ClientSession() as sess:

        # ① Upbit KRW/USDT (기준 데이터)
        all_usdt = []
        if mode == "agent":
            url = f"{BASE_UPBIT}/candles/days"
            params = {"market": "KRW-USDT", "count": lookback}
            async with sess.get(url, params=params) as resp:
                data = await resp.json()
            data = sorted(data, key=lambda x: x["candle_date_time_utc"])
            for d in data:
                ts = datetime.fromisoformat(d["candle_date_time_utc"]).replace(tzinfo=utc)
                all_usdt.append({
                    "timestamp_utc": ts,
                    "timestamp_kst": ts.astimezone(kst),
                    "close": float(d["trade_price"])
                })
        else:
            print(f"🚀 Start full data collection: Upbit KRW/USDT, K-premium (days)")
            end = datetime.now(kst)
            last_oldest = None
            while True:
                url = f"{BASE_UPBIT}/candles/days"
                params = {"market": "KRW-USDT", "count": 200, "to": end.strftime("%Y-%m-%d %H:%M:%S")}
                async with sess.get(url, params=params) as resp:
                    data = await resp.json()
                if not data:
                    print("✅ No more data, stopping.")
                    break
                data = sorted(data, key=lambda x: x["candle_date_time_utc"])
                for d in data:
                    ts = datetime.fromisoformat(d["candle_date_time_utc"]).replace(tzinfo=utc)
                    all_usdt.append({
                        "timestamp_utc": ts,
                        "timestamp_kst": ts.astimezone(kst),
                        "close": float(d["trade_price"])
                    })
                oldest = datetime.fromisoformat(data[-1]["candle_date_time_kst"])
                if last_oldest and abs((last_oldest - oldest).total_seconds()) < 1:
                    break
                last_oldest = oldest
                end = oldest - timedelta(days=200)
                print(f"📈 총 {len(all_usdt)}봉, oldest={oldest.date()}")
                await asyncio.sleep(0.25)

        if not all_usdt:
            print("⚠️ Upbit KRW/USDT 데이터 없음.")
            return pd.DataFrame()

        upbit_usdt = pd.DataFrame(all_usdt)
        upbit_usdt["date"] = upbit_usdt["timestamp_utc"].dt.date
        usdt_min, usdt_max = upbit_usdt["date"].min(), upbit_usdt["date"].max()

        # ② Upbit BTC/KRW
        all_btc = []
        end = datetime.now(kst)
        last_oldest = None
        while True:
            url = f"{BASE_UPBIT}/candles/days"
            params = {"market": "KRW-BTC", "count": 200, "to": end.strftime("%Y-%m-%d %H:%M:%S")}
            async with sess.get(url, params=params) as resp:
                data = await resp.json()
            if not data:
                break
            data = sorted(data, key=lambda x: x["candle_date_time_utc"])
            for d in data:
                ts = datetime.fromisoformat(d["candle_date_time_utc"]).replace(tzinfo=utc)
                all_btc.append({
                    "timestamp_utc": ts,
                    "timestamp_kst": ts.astimezone(kst),
                    "close": float(d["trade_price"])
                })
            oldest = datetime.fromisoformat(data[-1]["candle_date_time_kst"])
            if last_oldest and abs((last_oldest - oldest).total_seconds()) < 1:
                break
            if oldest.date() < usdt_min:
                break
            last_oldest = oldest
            end = oldest - timedelta(days=200)
            await asyncio.sleep(0.25)

        upbit_btc = pd.DataFrame(all_btc)
        upbit_btc["date"] = upbit_btc["timestamp_utc"].dt.date
        upbit_btc = upbit_btc[(upbit_btc["date"] >= usdt_min) & (upbit_btc["date"] <= usdt_max)]

        # ③ Binance BTC/USDT
        url_binance = f"{BASE_BINANCE}/klines"
        params_bin = {"symbol": "BTCUSDT", "interval": "1d", "limit": (usdt_max - usdt_min).days + 1}
        async with sess.get(url_binance, params=params_bin) as resp:
            data_bin = await resp.json()
        binance_btc = pd.DataFrame([
            {"timestamp_utc": datetime.utcfromtimestamp(d[0] / 1000).replace(tzinfo=utc), "close": float(d[4])}
            for d in data_bin
        ])
        binance_btc["date"] = binance_btc["timestamp_utc"].dt.date
        binance_btc = binance_btc[(binance_btc["date"] >= usdt_min) & (binance_btc["date"] <= usdt_max)]

        # ④ NBP 환율
        nbp_fx = await fetch_usdkrw_timeseries(start_date=usdt_min, end_date=usdt_max)

    # ✅ 타입 통일
    for df in [upbit_btc, binance_btc, upbit_usdt, nbp_fx]:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # ✅ 병합 및 계산
    merged = (
        upbit_btc
        .merge(binance_btc.add_prefix("binance_"), left_on="date", right_on="binance_date", how="inner")
        .merge(upbit_usdt.add_prefix("usdt_"), left_on="date", right_on="usdt_date", how="inner")
        .merge(nbp_fx, on="date", how="left")
        .sort_values("date").reset_index(drop=True)
    )
    merged["usdkrw"] = merged["usdkrw"].ffill().bfill()
    merged.rename(columns={
        "close": "upbit_btc_krw",
        "binance_close": "binance_btc_usdt",
        "usdt_close": "upbit_usdt_krw",
    }, inplace=True)
    merged["kimchi_premium(%)"] = (
        (merged["upbit_btc_krw"] - merged["binance_btc_usdt"] * merged["usdkrw"])
        / (merged["binance_btc_usdt"] * merged["usdkrw"])
    ) * 100

    # ✅ 불필요한 중간 컬럼 정리
    merged = merged.drop(columns=[
        "binance_timestamp_utc", "binance_date",
        "usdt_timestamp_utc", "usdt_timestamp_kst", "usdt_date", "date"
    ], errors="ignore")

    if save_csv:
        today = datetime.now().strftime("%Y%m%d")
        name = f"USDT_kimchi_days_{today}.csv"
        merged.to_csv(name, index=False)
        print(f"💾 Saved: {name} ({len(merged)} rows)")

    return merged


# ------------------------------------------------------------
# 실행 예시
# ------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        # ✅ [1] KRW-USDT 15분봉 (Agent)
        df15_agent = await fetch_usdt_candles(interval="minutes/15", mode="agent", lookback=15)
        print(f"📊 15분봉 (최근) 수집 완료: {len(df15_agent)} rows")

        # ✅ [2] KRW-USDT 15분봉 (Full + CSV)
        df15_full = await fetch_usdt_candles(interval="minutes/15", mode="full", save_csv=True)
        print(f"💾 15분봉 전체 저장 완료: {len(df15_full)} rows")

        # ✅ [3] KRW-USDT 1일봉 + 김프 (Agent)
        df_kimchi_agent = await fetch_kimchi_daily(mode="agent", lookback=30)
        print(f"📈 김프(Agent) 계산 완료: {len(df_kimchi_agent)} rows")

        # ✅ [4] KRW-USDT 1일봉 + 김프 (Full + CSV)
        df_kimchi_full = await fetch_kimchi_daily(mode="full", save_csv=True)
        print(f"💾 김프(Full) CSV 저장 완료: {len(df_kimchi_full)} rows")

    asyncio.run(main())
