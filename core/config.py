import os
import logging
from dotenv import load_dotenv
from loguru import logger
import ssl, certifi

# .env 파일 로드
load_dotenv()

# 로그 파일 저장 및 회전 설정
logger.add(
    "logs/trading_system.log",
    rotation="10 MB",        # 10MB 단위로 새 파일 생성
    retention="7 days",      # 7일 보관
    compression="zip",       # 오래된 로그는 압축
    level="INFO"
)

"""
해당 클래스에서 거래소 관련 설정, API 키, 거래 파라미터 등을 관리
"""
class Config:
    # API Keys (환경변수에서 불러오기)
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

    # Trading Parameters
    MAX_POSITION_SIZE = 100_000_000  # 1억 KRW
    MIN_ORDER_SIZE = 5000            # 최소 주문
    MAKER_FEE = 0.0005               # 메이커 수수료 (0.05%)
    TAKER_FEE = 0.0007               # 테이커 수수료 (0.07%)

    # Database -> SQLite 경로임
    DB_PATH = os.getenv("DB_PATH", "trading_data.db")

    # Logging 레벨 (기본 Info)
    LOG_LEVEL = logging.INFO

    # Rate Limiting (초당 최대 요청 수)
    MAX_REQUESTS_PER_SECOND = 10