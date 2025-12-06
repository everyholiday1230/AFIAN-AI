// Bybit Order Execution Engine
// 초저지연 주문 실행 시스템

use anyhow::{Context, Result};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum OrderType {
    Market,
    Limit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum TimeInForce {
    GTC,  // Good Till Cancel
    IOC,  // Immediate or Cancel
    FOK,  // Fill or Kill
    PostOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub reduce_only: bool,
    pub close_on_trigger: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub order_id: String,
    pub filled_quantity: f64,
    pub avg_price: f64,
    pub execution_time_ms: u64,
    pub slippage_bps: f64,
    pub fee: f64,
    pub status: OrderStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum OrderStatus {
    Created,
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

// Bybit API Request
#[derive(Debug, Serialize)]
struct BybitOrderRequest {
    category: String,
    symbol: String,
    side: String,
    #[serde(rename = "orderType")]
    order_type: String,
    qty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    price: Option<String>,
    #[serde(rename = "timeInForce")]
    time_in_force: String,
    #[serde(rename = "orderLinkId")]
    order_link_id: String,
    #[serde(rename = "reduceOnly")]
    reduce_only: bool,
    #[serde(rename = "closeOnTrigger")]
    close_on_trigger: bool,
}

// Bybit API Response
#[derive(Debug, Deserialize)]
struct BybitApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
    time: u64,
}

#[derive(Debug, Deserialize)]
struct BybitOrderResult {
    #[serde(rename = "orderId")]
    order_id: String,
    #[serde(rename = "orderLinkId")]
    order_link_id: String,
}

// ============================================================================
// Bybit Client
// ============================================================================

pub struct BybitClient {
    api_key: String,
    api_secret: String,
    base_url: String,
    client: Client,
    recv_window: u64,
}

impl BybitClient {
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let base_url = if testnet {
            "https://api-testnet.bybit.com".to_string()
        } else {
            "https://api.bybit.com".to_string()
        };

        Self {
            api_key,
            api_secret,
            base_url,
            client: Client::new(),
            recv_window: 5000,
        }
    }

    fn generate_signature(&self, timestamp: u64, params: &str) -> String {
        let sign_str = format!("{}{}{}", timestamp, &self.api_key, self.recv_window);
        let sign_str = if params.is_empty() {
            sign_str
        } else {
            format!("{}{}", sign_str, params)
        };

        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(sign_str.as_bytes());
        
        hex::encode(mac.finalize().into_bytes())
    }

    pub async fn place_order(&self, order: &Order) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        
        // Prepare request
        let order_request = BybitOrderRequest {
            category: "linear".to_string(),  // USDT perpetual
            symbol: order.symbol.clone(),
            side: match order.side {
                OrderSide::Buy => "Buy".to_string(),
                OrderSide::Sell => "Sell".to_string(),
            },
            order_type: match order.order_type {
                OrderType::Market => "Market".to_string(),
                OrderType::Limit => "Limit".to_string(),
            },
            qty: order.quantity.to_string(),
            price: order.price.map(|p| p.to_string()),
            time_in_force: match order.time_in_force {
                TimeInForce::GTC => "GTC".to_string(),
                TimeInForce::IOC => "IOC".to_string(),
                TimeInForce::FOK => "FOK".to_string(),
                TimeInForce::PostOnly => "PostOnly".to_string(),
            },
            order_link_id: order.id.clone(),
            reduce_only: order.reduce_only,
            close_on_trigger: order.close_on_trigger,
        };

        let body = serde_json::to_string(&order_request)?;
        let timestamp = Utc::now().timestamp_millis() as u64;
        let signature = self.generate_signature(timestamp, &body);

        // Send request
        let url = format!("{}/v5/order/create", self.base_url);
        
        let response = self.client
            .post(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-SIGN", signature)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-RECV-WINDOW", self.recv_window.to_string())
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Bybit API error: {}", error_text));
        }

        let api_response: BybitApiResponse<BybitOrderResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow::anyhow!(
                "Bybit API error: {} (code: {})",
                api_response.ret_msg,
                api_response.ret_code
            ));
        }

        let result = api_response.result.context("No result in response")?;

        // Query order details for fill information
        let order_info = self.get_order_info(&order.symbol, &result.order_id).await?;

        Ok(ExecutionResult {
            order_id: result.order_id,
            filled_quantity: order_info.filled_quantity,
            avg_price: order_info.avg_price,
            execution_time_ms,
            slippage_bps: order_info.slippage_bps,
            fee: order_info.fee,
            status: order_info.status,
        })
    }

    async fn get_order_info(&self, symbol: &str, order_id: &str) -> Result<OrderInfo> {
        // Simplified - in production, implement full order query
        // This would use GET /v5/order/realtime endpoint
        
        Ok(OrderInfo {
            filled_quantity: 0.0,  // Would be filled from API
            avg_price: 0.0,
            slippage_bps: 0.0,
            fee: 0.0,
            status: OrderStatus::New,
        })
    }

    pub async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<()> {
        let timestamp = Utc::now().timestamp_millis() as u64;
        
        let params = format!(
            r#"{{"category":"linear","symbol":"{}","orderId":"{}"}}"#,
            symbol, order_id
        );
        
        let signature = self.generate_signature(timestamp, &params);
        let url = format!("{}/v5/order/cancel", self.base_url);

        let response = self.client
            .post(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-SIGN", signature)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-RECV-WINDOW", self.recv_window.to_string())
            .header("Content-Type", "application/json")
            .body(params)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to cancel order"));
        }

        Ok(())
    }
}

#[derive(Debug)]
struct OrderInfo {
    filled_quantity: f64,
    avg_price: f64,
    slippage_bps: f64,
    fee: f64,
    status: OrderStatus,
}

// ============================================================================
// Smart Order Router
// ============================================================================

pub struct SmartOrderRouter {
    bybit_client: BybitClient,
    slippage_predictor: Arc<RwLock<SlippagePredictor>>,
    metrics: Arc<ExecutionMetrics>,
}

impl SmartOrderRouter {
    pub fn new(
        api_key: String,
        api_secret: String,
        testnet: bool,
    ) -> Self {
        Self {
            bybit_client: BybitClient::new(api_key, api_secret, testnet),
            slippage_predictor: Arc::new(RwLock::new(SlippagePredictor::new())),
            metrics: Arc::new(ExecutionMetrics::new()),
        }
    }

    pub async fn route_order(&self, order: Order) -> Result<ExecutionResult> {
        let start_time = Instant::now();

        info!(
            "🎯 Routing order: {} {} {} @ {:?}",
            order.symbol, 
            match order.side {
                OrderSide::Buy => "BUY",
                OrderSide::Sell => "SELL",
            },
            order.quantity,
            order.price
        );

        // 1. Predict slippage
        let predictor = self.slippage_predictor.read().await;
        let predicted_slippage = predictor.predict(&order).await?;
        drop(predictor);

        info!("📊 Predicted slippage: {:.2} bps", predicted_slippage * 10000.0);

        // 2. Adjust order if needed based on slippage
        let adjusted_order = self.adjust_order_for_slippage(order, predicted_slippage);

        // 3. Execute order
        let execution_result = self.bybit_client.place_order(&adjusted_order).await?;

        let total_latency = start_time.elapsed();

        // 4. Record metrics
        self.metrics.record_execution(
            &execution_result,
            total_latency,
            predicted_slippage,
        ).await;

        info!(
            "✅ Order executed: {} @ {} (latency: {:?})",
            execution_result.filled_quantity,
            execution_result.avg_price,
            total_latency
        );

        Ok(execution_result)
    }

    fn adjust_order_for_slippage(&self, mut order: Order, predicted_slippage: f64) -> Order {
        // For market orders, we can't adjust price
        // For limit orders, adjust limit price based on predicted slippage
        
        if let (OrderType::Limit, Some(price)) = (order.order_type.clone(), order.price) {
            let slippage_adjustment = price * predicted_slippage;
            
            order.price = Some(match order.side {
                OrderSide::Buy => price + slippage_adjustment,   // Buy higher
                OrderSide::Sell => price - slippage_adjustment,  // Sell lower
            });
        }

        order
    }

    pub async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<()> {
        self.bybit_client.cancel_order(symbol, order_id).await
    }
}

// ============================================================================
// Slippage Predictor
// ============================================================================

pub struct SlippagePredictor {
    // In production, this would use an ONNX model
    // For now, simple heuristic
}

impl SlippagePredictor {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn predict(&self, order: &Order) -> Result<f64> {
        // Simple heuristic: slippage increases with order size
        // In production, use ML model with market features
        
        let base_slippage = 0.0001;  // 1 bps
        let size_factor = (order.quantity / 1.0).ln().max(1.0);
        
        let predicted_slippage = base_slippage * size_factor;
        
        Ok(predicted_slippage)
    }
}

// ============================================================================
// Execution Metrics
// ============================================================================

pub struct ExecutionMetrics {
    total_orders: Arc<RwLock<u64>>,
    total_latency_ms: Arc<RwLock<f64>>,
    total_slippage: Arc<RwLock<f64>>,
}

impl ExecutionMetrics {
    pub fn new() -> Self {
        Self {
            total_orders: Arc::new(RwLock::new(0)),
            total_latency_ms: Arc::new(RwLock::new(0.0)),
            total_slippage: Arc::new(RwLock::new(0.0)),
        }
    }

    pub async fn record_execution(
        &self,
        result: &ExecutionResult,
        latency: Duration,
        predicted_slippage: f64,
    ) {
        let mut orders = self.total_orders.write().await;
        let mut total_lat = self.total_latency_ms.write().await;
        let mut total_slip = self.total_slippage.write().await;

        *orders += 1;
        *total_lat += latency.as_millis() as f64;
        *total_slip += result.slippage_bps;

        if *orders % 100 == 0 {
            let avg_latency = *total_lat / *orders as f64;
            let avg_slippage = *total_slip / *orders as f64;
            
            info!(
                "📈 Execution Stats: {} orders | Avg latency: {:.2}ms | Avg slippage: {:.2} bps",
                orders, avg_latency, avg_slippage
            );
        }
    }

    pub async fn get_stats(&self) -> (u64, f64, f64) {
        let orders = *self.total_orders.read().await;
        let total_lat = *self.total_latency_ms.read().await;
        let total_slip = *self.total_slippage.read().await;

        let avg_latency = if orders > 0 {
            total_lat / orders as f64
        } else {
            0.0
        };

        let avg_slippage = if orders > 0 {
            total_slip / orders as f64
        } else {
            0.0
        };

        (orders, avg_latency, avg_slippage)
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("order_executor=info")
        .with_target(false)
        .with_thread_ids(true)
        .init();

    info!("
    ╔═══════════════════════════════════════════════════════╗
    ║   QUANTUM ALPHA - Order Executor v0.1.0              ║
    ║   Ultra-Low Latency Bybit Execution Engine           ║
    ╚═══════════════════════════════════════════════════════╝
    ");

    // Get API credentials from environment
    let api_key = std::env::var("BYBIT_API_KEY")
        .unwrap_or_else(|_| {
            warn!("⚠️  BYBIT_API_KEY not set, using test mode");
            "test_key".to_string()
        });

    let api_secret = std::env::var("BYBIT_API_SECRET")
        .unwrap_or_else(|_| {
            warn!("⚠️  BYBIT_API_SECRET not set, using test mode");
            "test_secret".to_string()
        });

    let testnet = std::env::var("BYBIT_TESTNET")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);

    info!("🔧 Configuration:");
    info!("   API Key: {}...", &api_key[..8.min(api_key.len())]);
    info!("   Testnet: {}", testnet);

    // Create order router
    let router = SmartOrderRouter::new(api_key, api_secret, testnet);

    info!("✅ Order Executor initialized");
    info!("🎯 Ready to execute orders");

    // Example: Place a test order (commented out for safety)
    /*
    let test_order = Order {
        id: Uuid::new_v4().to_string(),
        symbol: "BTCUSDT".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        quantity: 0.001,
        price: Some(40000.0),
        time_in_force: TimeInForce::GTC,
        reduce_only: false,
        close_on_trigger: false,
    };

    match router.route_order(test_order).await {
        Ok(result) => {
            info!("✅ Test order executed successfully: {:?}", result);
        }
        Err(e) => {
            error!("❌ Test order failed: {:?}", e);
        }
    }
    */

    // Keep running
    info!("📡 Order Executor running. Press Ctrl+C to stop.");
    
    tokio::signal::ctrl_c().await?;
    
    info!("🛑 Shutting down Order Executor...");

    // Print final stats
    let (orders, avg_lat, avg_slip) = router.metrics.get_stats().await;
    info!("
    📊 Final Statistics:
       Total Orders: {}
       Avg Latency: {:.2}ms
       Avg Slippage: {:.2} bps
    ", orders, avg_lat, avg_slip);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 0.001,
            price: None,
            time_in_force: TimeInForce::IOC,
            reduce_only: false,
            close_on_trigger: false,
        };

        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.quantity, 0.001);
    }

    #[tokio::test]
    async fn test_slippage_predictor() {
        let predictor = SlippagePredictor::new();
        
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 1.0,
            price: None,
            time_in_force: TimeInForce::IOC,
            reduce_only: false,
            close_on_trigger: false,
        };

        let slippage = predictor.predict(&order).await.unwrap();
        assert!(slippage > 0.0);
        assert!(slippage < 0.01); // Less than 100 bps
    }
}
