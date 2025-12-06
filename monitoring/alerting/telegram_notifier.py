"""
Telegram Alert System
실시간 트레이딩 알림 시스템
"""

import asyncio
import os
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed")


class TelegramNotifier:
    """Telegram 알림 시스템"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram notifications disabled - package not installed")
            self.enabled = False
            return
        
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not found")
            self.enabled = False
        else:
            self.bot = Bot(token=self.bot_token)
            self.enabled = True
            logger.success("Telegram notifier initialized")
    
    async def send_message(self, message: str, parse_mode: str = 'HTML'):
        """메시지 전송"""
        if not self.enabled:
            return
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    async def notify_trade(self, trade: Dict):
        """거래 알림"""
        message = f"""
🔔 <b>Trade Executed</b>

Symbol: {trade.get('symbol', 'N/A')}
Side: {trade.get('side', 'N/A').upper()}
Price: ${trade.get('price', 0):.2f}
Size: {trade.get('size', 0):.4f}
PnL: ${trade.get('pnl', 0):.2f}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)
    
    async def notify_risk_alert(self, alert: Dict):
        """리스크 알림"""
        message = f"""
⚠️ <b>RISK ALERT</b>

Type: {alert.get('type', 'Unknown')}
Level: {alert.get('level', 'WARNING')}
Message: {alert.get('message', 'N/A')}

Current Equity: ${alert.get('equity', 0):.2f}
Drawdown: {alert.get('drawdown_pct', 0):.2f}%

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)
    
    async def notify_system_status(self, status: Dict):
        """시스템 상태 알림"""
        message = f"""
📊 <b>System Status</b>

Status: {status.get('status', 'UNKNOWN')}
Uptime: {status.get('uptime', 'N/A')}
Total Trades: {status.get('total_trades', 0)}
Win Rate: {status.get('win_rate', 0):.2f}%

Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)


if __name__ == "__main__":
    async def test():
        notifier = TelegramNotifier()
        
        if notifier.enabled:
            await notifier.notify_system_status({
                'status': 'RUNNING',
                'uptime': '2h 30m',
                'total_trades': 42,
                'win_rate': 62.5,
            })
            logger.success("✅ Test notification sent")
        else:
            logger.warning("Notifications disabled in test mode")
    
    asyncio.run(test())
