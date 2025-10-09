import aiohttp, logging, uuid
from datetime import datetime
from typing import Dict, Any
from config import Config
from model import OrderResult


class UpbitExecutionEngine:
    def __init__(self, access_key=None, secret_key=None):
        self.access_key = access_key or Config.UPBIT_ACCESS_KEY
        self.secret_key = secret_key or Config.UPBIT_SECRET_KEY
        self.base_url = "https://api.upbit.com/v1"
        self.session = None
        self.pending_orders = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=Config.SSL_CONTEXT)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _calc_fee(self, amount: float, price: float, taker=False) -> float:
        fee_rate = Config.TAKER_FEE if taker else Config.MAKER_FEE
        return amount * price * fee_rate

    async def execute_order(self, action: Dict[str, Any]) -> OrderResult:
        side = action.get('action')
        size_ratio = action.get('size', 0.1)
        # 잔고 확인, 주문량 계산, 오더북에서 최적가격 찾는 부분 동일
        # 체결 결과 만들 때 수수료 반영:
        filled_amount = 100  # 예시
        filled_price = 1300  # 예시
        fee = self._calc_fee(filled_amount, filled_price, taker=False)
        return OrderResult(
            order_id=str(uuid.uuid4()),
            status="filled",
            filled_amount=filled_amount,
            filled_price=filled_price,
            fee=fee,
            timestamp=datetime.now()
        )