"""
Email Notification System

목적: 중요한 트레이딩 이벤트를 이메일로 알림

주요 알림:
- 일일/주간 성과 리포트
- Kill Switch 발동 알림  
- 시스템 크리티컬 에러
- 대규모 손익 이벤트
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmailNotifier:
    """
    SMTP 이메일 알림 시스템
    
    Args:
        smtp_server: SMTP 서버 주소
        smtp_port: SMTP 포트
        sender_email: 발신자 이메일
        sender_password: 발신자 비밀번호
        receiver_emails: 수신자 이메일 리스트
        use_tls: TLS 사용 여부
    """
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        receiver_emails: List[str],
        use_tls: bool = True
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_emails = receiver_emails
        self.use_tls = use_tls
    
    def send_email(
        self,
        subject: str,
        body_html: str,
        body_plain: Optional[str] = None
    ):
        """
        이메일 전송
        
        Args:
            subject: 제목
            body_html: HTML 본문
            body_plain: Plain text 본문 (선택사항)
        """
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = f"[Quantum Alpha] {subject}"
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.receiver_emails)
            
            # Plain text 버전
            if body_plain:
                part1 = MIMEText(body_plain, "plain")
                message.attach(part1)
            
            # HTML 버전
            part2 = MIMEText(body_html, "html")
            message.attach(part2)
            
            # SMTP 연결 및 전송
            context = ssl.create_default_context() if self.use_tls else None
            
            if self.use_tls:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(
                        self.sender_email,
                        self.receiver_emails,
                        message.as_string()
                    )
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(
                        self.sender_email,
                        self.receiver_emails,
                        message.as_string()
                    )
            
            logger.info(f"Email sent successfully: {subject}")
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def notify_daily_report(
        self,
        date: str,
        total_trades: int,
        win_rate: float,
        total_pnl: float,
        sharpe_ratio: float,
        max_drawdown: float,
        top_trades: List[dict]
    ):
        """일일 성과 리포트"""
        subject = f"Daily Report - {date}"
        
        # HTML 본문
        html = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; }}
              .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
              .content {{ padding: 20px; }}
              .metric {{ margin: 10px 0; }}
              .metric-label {{ font-weight: bold; color: #34495e; }}
              .metric-value {{ color: #2c3e50; font-size: 18px; }}
              .positive {{ color: #27ae60; }}
              .negative {{ color: #e74c3c; }}
              table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
              th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
              th {{ background-color: #34495e; color: white; }}
            </style>
          </head>
          <body>
            <div class="header">
              <h1>🚀 Quantum Alpha Daily Report</h1>
              <p>{date}</p>
            </div>
            <div class="content">
              <h2>📊 Performance Summary</h2>
              
              <div class="metric">
                <span class="metric-label">Total Trades:</span>
                <span class="metric-value">{total_trades}</span>
              </div>
              
              <div class="metric">
                <span class="metric-label">Win Rate:</span>
                <span class="metric-value">{win_rate*100:.1f}%</span>
              </div>
              
              <div class="metric">
                <span class="metric-label">Total PnL:</span>
                <span class="metric-value {'positive' if total_pnl > 0 else 'negative'}">${total_pnl:+,.2f}</span>
              </div>
              
              <div class="metric">
                <span class="metric-label">Sharpe Ratio:</span>
                <span class="metric-value">{sharpe_ratio:.2f}</span>
              </div>
              
              <div class="metric">
                <span class="metric-label">Max Drawdown:</span>
                <span class="metric-value negative">{max_drawdown*100:.2f}%</span>
              </div>
              
              <h2>🏆 Top Trades</h2>
              <table>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Price</th>
                  <th>Quantity</th>
                  <th>PnL</th>
                </tr>
        """
        
        for trade in top_trades[:5]:
            pnl_class = 'positive' if trade['pnl'] > 0 else 'negative'
            html += f"""
                <tr>
                  <td>{trade['symbol']}</td>
                  <td>{trade['side']}</td>
                  <td>${trade['price']:,.2f}</td>
                  <td>{trade['quantity']:.4f}</td>
                  <td class="{pnl_class}">${trade['pnl']:+,.2f}</td>
                </tr>
            """
        
        html += """
              </table>
            </div>
          </body>
        </html>
        """
        
        # Plain text 버전
        plain = f"""
Quantum Alpha Daily Report - {date}

Performance Summary:
- Total Trades: {total_trades}
- Win Rate: {win_rate*100:.1f}%
- Total PnL: ${total_pnl:+,.2f}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown*100:.2f}%

Top Trades:
"""
        for i, trade in enumerate(top_trades[:5], 1):
            plain += f"{i}. {trade['symbol']} {trade['side']} @ ${trade['price']:,.2f} - PnL: ${trade['pnl']:+,.2f}\\n"
        
        self.send_email(subject, html, plain)
    
    def notify_kill_switch(
        self,
        reason: str,
        details: dict
    ):
        """Kill Switch 발동 알림"""
        subject = "🚨 KILL SWITCH ACTIVATED"
        
        html = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; }}
              .header {{ background-color: #c0392b; color: white; padding: 20px; text-align: center; }}
              .content {{ padding: 20px; }}
              .alert {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; margin: 10px 0; }}
            </style>
          </head>
          <body>
            <div class="header">
              <h1>🚨 KILL SWITCH ACTIVATED</h1>
            </div>
            <div class="content">
              <div class="alert">
                <h2>Reason: {reason}</h2>
              </div>
              
              <h3>Details:</h3>
              <ul>
        """
        
        for key, value in details.items():
            html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
        
        html += """
              </ul>
              
              <p><strong>Action Required:</strong> Please review the system immediately and investigate the cause.</p>
            </div>
          </body>
        </html>
        """
        
        plain = f"""
KILL SWITCH ACTIVATED

Reason: {reason}

Details:
"""
        for key, value in details.items():
            plain += f"- {key.replace('_', ' ').title()}: {value}\\n"
        
        plain += "\\nAction Required: Please review the system immediately."
        
        self.send_email(subject, html, plain)


if __name__ == "__main__":
    print("🧪 Testing Email Notifier...")
    
    # 테스트용 설정 (실제 환경에서는 환경 변수로 관리)
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 465
    SENDER_EMAIL = "your_email@gmail.com"
    SENDER_PASSWORD = "your_app_password"
    RECEIVER_EMAILS = ["receiver@example.com"]
    
    if SENDER_EMAIL == "your_email@gmail.com":
        print("⚠️  Please configure email settings to test")
        print("   For Gmail:")
        print("   1. Enable 2-factor authentication")
        print("   2. Generate App Password")
        print("   3. Use App Password instead of regular password")
    else:
        notifier = EmailNotifier(
            smtp_server=SMTP_SERVER,
            smtp_port=SMTP_PORT,
            sender_email=SENDER_EMAIL,
            sender_password=SENDER_PASSWORD,
            receiver_emails=RECEIVER_EMAILS
        )
        
        # 일일 리포트 테스트
        notifier.notify_daily_report(
            date="2024-01-15",
            total_trades=25,
            win_rate=0.64,
            total_pnl=320.50,
            sharpe_ratio=2.8,
            max_drawdown=0.08,
            top_trades=[
                {"symbol": "BTCUSDT", "side": "BUY", "price": 50000, "quantity": 0.1, "pnl": 150.50},
                {"symbol": "ETHUSDT", "side": "SELL", "price": 3000, "quantity": 1.0, "pnl": 80.20},
            ]
        )
    
    print("✅ Email Notifier test completed!")
