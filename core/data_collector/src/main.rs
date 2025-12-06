use tokio_tungstenite::{connect_async, tungstenite::Message};
use serde::{Deserialize, Serialize};
use redis::AsyncCommands;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::{Result, Context};
use tracing::{info, warn, error};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MarketTick {
    pub timestamp: u64,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: OrderSide,
    pub trade_id: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct CollectorMetrics {
    pub total_messages: Arc<Mutex<u64>>,
    pub total_latency_ms: Arc<Mutex<f64>>,
    pub error_count: Arc<Mutex<u64>>,
}

impl CollectorMetrics {
    pub fn new() -> Self {
        Self {
            total_messages: Arc::new(Mutex::new(0)),
            total_latency_ms: Arc::new(Mutex::new(0.0)),
            error_count: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn record_latency(&self, latency: Duration) {
        let mut total_msgs = self.total_messages.lock().await;
        let mut total_lat = self.total_latency_ms.lock().await;
        
        *total_msgs += 1;
        *total_lat += latency.as_secs_f64() * 1000.0;
        
        if *total_msgs % 1000 == 0 {
            let avg_latency = *total_lat / *total_msgs as f64;
            info!("📊 Avg latency: {:.2}ms | Total messages: {}", avg_latency, total_msgs);
        }
    }

    pub async fn record_error(&self) {
        let mut errors = self.error_count.lock().await;
        *errors += 1;
    }
}

pub struct DataCollector {
    redis_url: String,
    buffer: Arc<lockfree::queue::Queue<MarketTick>>,
    metrics: CollectorMetrics,
}

impl DataCollector {
    pub fn new(redis_url: String) -> Self {
        Self {
            redis_url,
            buffer: Arc::new(lockfree::queue::Queue::new()),
            metrics: CollectorMetrics::new(),
        }
    }

    pub async fn start_collection(&self) -> Result<()> {
        info!("🚀 Starting Quantum Alpha Data Collector");
        
        // Binance Futures 거래소 연결
        let exchanges = vec![
            ("binance", "wss://fstream.binance.com/ws/btcusdt@trade"),
            ("binance", "wss://fstream.binance.com/ws/ethusdt@trade"),
        ];
        
        let mut handles = vec![];
        
        // 각 거래소별 수집 태스크 생성
        for (exchange_name, exchange_url) in exchanges {
            let buffer_clone = Arc::clone(&self.buffer);
            let metrics_clone = self.metrics.clone();
            let exchange_name = exchange_name.to_string();
            let exchange_url = exchange_url.to_string();
            
            let handle = tokio::spawn(async move {
                Self::collect_from_exchange(
                    &exchange_name,
                    &exchange_url,
                    buffer_clone,
                    metrics_clone,
                )
                .await
            });
            
            handles.push(handle);
        }
        
        // 데이터 처리 루프
        let buffer_clone = Arc::clone(&self.buffer);
        let redis_url = self.redis_url.clone();
        let process_handle = tokio::spawn(async move {
            Self::process_data_loop(buffer_clone, redis_url).await
        });
        handles.push(process_handle);
        
        // 모든 태스크 완료 대기
        for handle in handles {
            if let Err(e) = handle.await {
                error!("❌ Task failed: {:?}", e);
            }
        }
        
        Ok(())
    }
    
    async fn collect_from_exchange(
        exchange_name: &str,
        url: &str,
        buffer: Arc<lockfree::queue::Queue<MarketTick>>,
        metrics: CollectorMetrics,
    ) -> Result<()> {
        info!("🔌 Connecting to {} at {}", exchange_name, url);
        
        loop {
            match connect_async(url).await {
                Ok((ws_stream, _)) => {
                    info!("✅ Connected to {}", exchange_name);
                    
                    let (_, mut read) = ws_stream.split();
                    
                    while let Some(message) = {
                        use futures::StreamExt;
                        read.next().await
                    } {
                        let start_time = Instant::now();
                        
                        match message {
                            Ok(Message::Text(text)) => {
                                match Self::parse_binance_data(&text) {
                                    Ok(tick) => {
                                        buffer.push(tick);
                                        metrics.record_latency(start_time.elapsed()).await;
                                    }
                                    Err(e) => {
                                        warn!("⚠️  Parse error: {:?}", e);
                                        metrics.record_error().await;
                                    }
                                }
                            }
                            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
                            Err(e) => {
                                error!("❌ WebSocket error: {:?}", e);
                                metrics.record_error().await;
                                break;
                            }
                            _ => {}
                        }
                    }
                    
                    warn!("⚠️  Connection lost to {}, reconnecting...", exchange_name);
                }
                Err(e) => {
                    error!("❌ Connection failed to {}: {:?}", exchange_name, e);
                    metrics.record_error().await;
                }
            }
            
            // 재연결 대기
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }
    
    fn parse_binance_data(text: &str) -> Result<MarketTick> {
        #[derive(Deserialize)]
        struct BinanceTradeMsg {
            #[serde(rename = "E")]
            event_time: u64,
            #[serde(rename = "s")]
            symbol: String,
            #[serde(rename = "p")]
            price: String,
            #[serde(rename = "q")]
            quantity: String,
            #[serde(rename = "m")]
            is_buyer_maker: bool,
            #[serde(rename = "t")]
            trade_id: u64,
        }
        
        let msg: BinanceTradeMsg = serde_json::from_str(text)
            .context("Failed to parse Binance message")?;
        
        Ok(MarketTick {
            timestamp: msg.event_time,
            symbol: msg.symbol,
            price: msg.price.parse().context("Failed to parse price")?,
            quantity: msg.quantity.parse().context("Failed to parse quantity")?,
            side: if msg.is_buyer_maker {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            },
            trade_id: msg.trade_id,
        })
    }
    
    async fn process_data_loop(
        buffer: Arc<lockfree::queue::Queue<MarketTick>>,
        redis_url: String,
    ) -> Result<()> {
        // Redis 연결
        let client = redis::Client::open(redis_url.as_str())
            .context("Failed to create Redis client")?;
        let mut con = client.get_async_connection().await
            .context("Failed to connect to Redis")?;
        
        info!("✅ Redis connected");
        
        let mut batch = Vec::with_capacity(1000);
        let mut last_log = Instant::now();
        
        loop {
            // 배치 수집 (최대 1ms 대기)
            let deadline = Instant::now() + Duration::from_millis(1);
            
            while Instant::now() < deadline && batch.len() < 1000 {
                if let Some(tick) = buffer.pop() {
                    batch.push(tick);
                }
            }
            
            if !batch.is_empty() {
                // Redis에 저장 (스트림 사용)
                for tick in &batch {
                    let key = format!("market:{}:trades", tick.symbol);
                    let json = serde_json::to_string(tick)?;
                    
                    let _: () = con.lpush(&key, json).await?;
                    let _: () = con.ltrim(&key, 0, 9999).await?; // 최근 10000개만 유지
                }
                
                // 주기적 로그
                if last_log.elapsed() > Duration::from_secs(10) {
                    info!("💾 Processed {} ticks", batch.len());
                    last_log = Instant::now();
                }
                
                batch.clear();
            }
            
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 로깅 초기화
    tracing_subscriber::fmt()
        .with_env_filter("data_collector=info")
        .with_target(false)
        .with_thread_ids(true)
        .init();
    
    info!("
    ╔═══════════════════════════════════════════════════════╗
    ║     QUANTUM ALPHA - Data Collector v0.1.0            ║
    ║     Ultra-Low Latency Market Data Engine             ║
    ╚═══════════════════════════════════════════════════════╝
    ");
    
    // Redis URL (환경변수 또는 기본값)
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    
    let collector = DataCollector::new(redis_url);
    
    // 데이터 수집 시작
    collector.start_collection().await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_binance_data() {
        let json = r#"{
            "e": "trade",
            "E": 1672515782136,
            "s": "BTCUSDT",
            "t": 12345,
            "p": "42000.50",
            "q": "0.01",
            "m": true
        }"#;
        
        let result = DataCollector::parse_binance_data(json);
        assert!(result.is_ok());
        
        let tick = result.unwrap();
        assert_eq!(tick.symbol, "BTCUSDT");
        assert_eq!(tick.price, 42000.50);
    }
}
