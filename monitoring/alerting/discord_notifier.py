"""
Discord Notification System

목적: 트레이딩 이벤트 및 시스템 알림을 Discord로 전송

주요 알림:
- 거래 체결 알림
- Kill Switch 발동 알림
- 시스템 에러 알림
- 성과 리포트 알림
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DiscordNotifier:
    """
    Discord Webhook 알림 시스템
    
    Args:
        webhook_url: Discord Webhook URL
        username: 봇 사용자 이름
        avatar_url: 봇 아바타 URL (선택사항)
    """
    
    # Discord 색상 코드
    COLORS = {
        AlertLevel.INFO: 3447003,      # 파란색
        AlertLevel.WARNING: 16776960,  # 노란색
        AlertLevel.ERROR: 15158332,    # 빨간색
        AlertLevel.CRITICAL: 10038562  # 진한 빨간색
    }
    
    # 이모지
    EMOJIS = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.ERROR: "❌",
        AlertLevel.CRITICAL: "🚨"
    }
    
    def __init__(
        self,
        webhook_url: str,
        username: str = "Quantum Alpha Bot",
        avatar_url: Optional[str] = None
    ):
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
    
    async def send_message(
        self,
        title: str,
        description: str,
        level: AlertLevel = AlertLevel.INFO,
        fields: Optional[List[Dict[str, str]]] = None,
        thumbnail_url: Optional[str] = None
    ):
        """
        Discord 메시지 전송
        
        Args:
            title: 메시지 제목
            description: 메시지 내용
            level: 알림 레벨
            fields: 추가 필드 [{"name": "...", "value": "...", "inline": True/False}]
            thumbnail_url: 썸네일 이미지 URL
        """
        try:
            embed = {
                "title": f"{self.EMOJIS[level]} {title}",
                "description": description,
                "color": self.COLORS[level],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {
                    "text": "Quantum Alpha Trading System"
                }
            }
            
            if fields:
                embed["fields"] = fields
            
            if thumbnail_url:
                embed["thumbnail"] = {"url": thumbnail_url}
            
            payload = {
                "username": self.username,
                "embeds": [embed]
            }
            
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 204:
                        logger.info(f"Discord notification sent: {title}")
                    else:
                        logger.error(f"Failed to send Discord notification: {response.status}")
        
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
    
    async def notify_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None
    ):
        """거래 체결 알림"""
        emoji = "📈" if side.upper() == "BUY" else "📉"
        
        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Side", "value": side.upper(), "inline": True},
            {"name": "Price", "value": f"${price:,.2f}", "inline": True},
            {"name": "Quantity", "value": f"{quantity:.4f}", "inline": True},
        ]
        
        if pnl is not None:
            pnl_emoji = "💰" if pnl > 0 else "💸"
            fields.append({
                "name": f"{pnl_emoji} PnL",
                "value": f"${pnl:+,.2f}",
                "inline": True
            })
        
        await self.send_message(
            title=f"{emoji} Trade Executed",
            description=f"New {side.lower()} order filled on {symbol}",
            level=AlertLevel.INFO,
            fields=fields
        )
    
    async def notify_kill_switch(
        self,
        reason: str,
        details: Dict[str, any]
    ):
        """Kill Switch 발동 알림"""
        fields = [
            {"name": "Reason", "value": reason, "inline": False},
        ]
        
        for key, value in details.items():
            fields.append({
                "name": key.replace("_", " ").title(),
                "value": str(value),
                "inline": True
            })
        
        await self.send_message(
            title="🚨 KILL SWITCH ACTIVATED",
            description="Trading has been halted due to risk management trigger",
            level=AlertLevel.CRITICAL,
            fields=fields
        )
    
    async def notify_error(
        self,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None
    ):
        """에러 알림"""
        description = f"**Error Type:** {error_type}\\n**Message:** {error_message}"
        
        if traceback:
            description += f"\\n\\n```\\n{traceback[:500]}\\n```"
        
        await self.send_message(
            title="System Error",
            description=description,
            level=AlertLevel.ERROR
        )
    
    async def notify_daily_report(
        self,
        date: str,
        total_trades: int,
        win_rate: float,
        pnl: float,
        sharpe_ratio: float
    ):
        """일일 성과 리포트"""
        pnl_emoji = "💰" if pnl > 0 else "💸"
        
        fields = [
            {"name": "Date", "value": date, "inline": False},
            {"name": "Total Trades", "value": str(total_trades), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate*100:.1f}%", "inline": True},
            {"name": f"{pnl_emoji} PnL", "value": f"${pnl:+,.2f}", "inline": True},
            {"name": "Sharpe Ratio", "value": f"{sharpe_ratio:.2f}", "inline": True},
        ]
        
        level = AlertLevel.INFO if pnl > 0 else AlertLevel.WARNING
        
        await self.send_message(
            title="📊 Daily Performance Report",
            description="Here's your trading performance for today",
            level=level,
            fields=fields
        )


def send_discord_notification_sync(
    webhook_url: str,
    title: str,
    description: str,
    level: AlertLevel = AlertLevel.INFO
):
    """동기 버전 (간단한 알림용)"""
    try:
        notifier = DiscordNotifier(webhook_url)
        asyncio.run(notifier.send_message(title, description, level))
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")


if __name__ == "__main__":
    print("🧪 Testing Discord Notifier...")
    
    # 테스트용 webhook URL (실제 환경에서는 환경 변수로 관리)
    WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"
    
    if WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print("⚠️  Please set your Discord webhook URL to test")
    else:
        notifier = DiscordNotifier(WEBHOOK_URL)
        
        async def test():
            # 거래 알림 테스트
            await notifier.notify_trade(
                symbol="BTCUSDT",
                side="BUY",
                price=50000.0,
                quantity=0.1,
                pnl=150.50
            )
            
            # Kill Switch 알림 테스트
            await notifier.notify_kill_switch(
                reason="Daily Loss Limit Exceeded",
                details={
                    "daily_pnl": -500.0,
                    "limit": -450.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # 일일 리포트 테스트
            await notifier.notify_daily_report(
                date="2024-01-15",
                total_trades=25,
                win_rate=0.64,
                pnl=320.50,
                sharpe_ratio=2.8
            )
        
        asyncio.run(test())
    
    print("✅ Discord Notifier test completed!")
