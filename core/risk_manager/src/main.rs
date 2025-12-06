// Risk Management System with Kill Switch
// 엔터프라이즈급 리스크 관리 및 긴급 정지 시스템

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLimit {
    pub max_position_size: f64,
    pub max_leverage: f64,
    pub max_risk_per_trade_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountState {
    pub balance: f64,
    pub equity: f64,
    pub margin_used: f64,
    pub positions: HashMap<String, Position>,
    pub daily_pnl: f64,
    pub consecutive_losses: u32,
    pub current_volatility: f64,
    pub api_error_rate: f64,
    pub model_disagreement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: PositionSide,
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub leverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    Neutral,
}

#[derive(Debug, Clone)]
pub enum KillSwitch {
    DailyLossLimit { threshold: f64 },
    ConsecutiveLosses { max_losses: u32 },
    VolatilitySpike { threshold: f64 },
    ApiErrorRate { threshold: f64 },
    ModelDisagreement { threshold: f64 },
    DrawdownLimit { max_drawdown_pct: f64 },
    LeverageExcess { max_leverage: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskViolation {
    PositionSizeExceeded,
    LeverageExceeded,
    DailyLossLimitExceeded,
    ConsecutiveLossesExceeded,
    VolatilitySpikeDetected,
    ApiErrorRateHigh,
    ModelDisagreementHigh,
    DrawdownExceeded,
    KillSwitchTriggered,
}

#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub is_approved: bool,
    pub violations: Vec<RiskViolation>,
    pub risk_score: f64,
    pub max_allowed_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub pnl: f64,
    pub return_pct: f64,
}

// ============================================================================
// Risk Manager
// ============================================================================

pub struct RiskManager {
    // Configuration
    position_limits: Arc<RwLock<HashMap<String, PositionLimit>>>,
    daily_loss_limit: f64,
    max_drawdown_pct: f64,
    kill_switches: Vec<KillSwitch>,
    
    // State
    is_trading_allowed: Arc<RwLock<bool>>,
    account_state: Arc<RwLock<AccountState>>,
    trade_history: Arc<RwLock<VecDeque<TradeRecord>>>,
    peak_equity: Arc<RwLock<f64>>,
    
    // Metrics
    total_risk_checks: Arc<RwLock<u64>>,
    total_violations: Arc<RwLock<u64>>,
}

impl RiskManager {
    pub fn new(
        daily_loss_limit: f64,
        max_drawdown_pct: f64,
        kill_switches: Vec<KillSwitch>,
    ) -> Self {
        Self {
            position_limits: Arc::new(RwLock::new(HashMap::new())),
            daily_loss_limit,
            max_drawdown_pct,
            kill_switches,
            is_trading_allowed: Arc::new(RwLock::new(true)),
            account_state: Arc::new(RwLock::new(AccountState::default())),
            trade_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            peak_equity: Arc::new(RwLock::new(0.0)),
            total_risk_checks: Arc::new(RwLock::new(0)),
            total_violations: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn set_position_limit(&self, symbol: String, limit: PositionLimit) {
        let mut limits = self.position_limits.write().await;
        limits.insert(symbol, limit);
    }

    pub async fn check_order_risk(
        &self,
        symbol: &str,
        side: &PositionSide,
        quantity: f64,
        price: f64,
    ) -> RiskCheckResult {
        let mut checks = self.total_risk_checks.write().await;
        *checks += 1;
        drop(checks);

        let mut violations = Vec::new();
        let account = self.account_state.read().await.clone();
        
        // Check if trading is allowed
        let trading_allowed = *self.is_trading_allowed.read().await;
        if !trading_allowed {
            violations.push(RiskViolation::KillSwitchTriggered);
            return RiskCheckResult {
                is_approved: false,
                violations,
                risk_score: 1.0,
                max_allowed_size: 0.0,
            };
        }

        // 1. Position size check
        if let Some(limit) = self.position_limits.read().await.get(symbol) {
            let current_position_size = account
                .positions
                .get(symbol)
                .map(|p| p.size.abs())
                .unwrap_or(0.0);
            
            let new_position_size = current_position_size + quantity;

            if new_position_size > limit.max_position_size {
                violations.push(RiskViolation::PositionSizeExceeded);
            }
        }

        // 2. Daily loss limit check
        if account.daily_pnl <= -self.daily_loss_limit {
            violations.push(RiskViolation::DailyLossLimitExceeded);
        }

        // 3. Leverage check
        let order_value = quantity * price;
        let new_margin_used = account.margin_used + order_value;
        let new_leverage = new_margin_used / account.equity;

        if new_leverage > 10.0 {
            violations.push(RiskViolation::LeverageExceeded);
        }

        // 4. Drawdown check
        let peak = *self.peak_equity.read().await;
        if peak > 0.0 {
            let current_drawdown = (peak - account.equity) / peak * 100.0;
            if current_drawdown > self.max_drawdown_pct {
                violations.push(RiskViolation::DrawdownExceeded);
            }
        }

        // 5. Kill switch checks
        for kill_switch in &self.kill_switches {
            if self.check_kill_switch_internal(kill_switch, &account).await {
                violations.push(RiskViolation::KillSwitchTriggered);
                self.emergency_shutdown().await;
                break;
            }
        }

        // Calculate risk score (0.0 = no risk, 1.0 = maximum risk)
        let risk_score = self.calculate_risk_score(&account, &violations);

        // Calculate max allowed size based on risk
        let max_allowed_size = self.calculate_max_allowed_size(
            symbol,
            &account,
            risk_score,
        ).await;

        let is_approved = violations.is_empty();

        if !is_approved {
            let mut total_violations = self.total_violations.write().await;
            *total_violations += 1;

            warn!(
                "⚠️  Risk check FAILED for {} {}: {:?}",
                symbol, quantity, violations
            );
        }

        RiskCheckResult {
            is_approved,
            violations,
            risk_score,
            max_allowed_size,
        }
    }

    async fn check_kill_switch_internal(
        &self,
        kill_switch: &KillSwitch,
        account: &AccountState,
    ) -> bool {
        match kill_switch {
            KillSwitch::DailyLossLimit { threshold } => {
                account.daily_pnl <= -threshold
            }
            KillSwitch::ConsecutiveLosses { max_losses } => {
                account.consecutive_losses >= *max_losses
            }
            KillSwitch::VolatilitySpike { threshold } => {
                account.current_volatility > *threshold
            }
            KillSwitch::ApiErrorRate { threshold } => {
                account.api_error_rate > *threshold
            }
            KillSwitch::ModelDisagreement { threshold } => {
                account.model_disagreement > *threshold
            }
            KillSwitch::DrawdownLimit { max_drawdown_pct } => {
                let peak = *self.peak_equity.read().await;
                if peak > 0.0 {
                    let current_drawdown = (peak - account.equity) / peak * 100.0;
                    current_drawdown > *max_drawdown_pct
                } else {
                    false
                }
            }
            KillSwitch::LeverageExcess { max_leverage } => {
                let leverage = account.margin_used / account.equity;
                leverage > *max_leverage
            }
        }
    }

    pub async fn emergency_shutdown(&self) {
        error!("🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨");

        // Stop all trading
        let mut trading_allowed = self.is_trading_allowed.write().await;
        *trading_allowed = false;

        info!("🛑 All trading halted");

        // In production:
        // 1. Close all positions (or implement position reduction strategy)
        // 2. Cancel all pending orders
        // 3. Send emergency alerts (email, SMS, Telegram, etc.)
        // 4. Log incident to database
        // 5. Notify operations team

        // Log the event
        error!(
            "Emergency shutdown at {}. Account state: equity=${:.2}, daily_pnl=${:.2}",
            Utc::now(),
            self.account_state.read().await.equity,
            self.account_state.read().await.daily_pnl
        );
    }

    pub async fn resume_trading(&self, authorization_code: &str) -> Result<()> {
        // In production, require multi-factor authorization
        if authorization_code == "EMERGENCY_RESUME_CODE" {
            let mut trading_allowed = self.is_trading_allowed.write().await;
            *trading_allowed = true;
            info!("✅ Trading resumed by authorized personnel");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid authorization code"))
        }
    }

    pub async fn update_account_state(&self, new_state: AccountState) {
        let mut state = self.account_state.write().await;
        *state = new_state.clone();

        // Update peak equity
        let mut peak = self.peak_equity.write().await;
        if new_state.equity > *peak {
            *peak = new_state.equity;
        }
    }

    pub async fn record_trade(&self, record: TradeRecord) {
        let mut history = self.trade_history.write().await;
        history.push_back(record.clone());

        // Keep only last 1000 trades
        if history.len() > 1000 {
            history.pop_front();
        }

        // Update consecutive losses
        if record.pnl < 0.0 {
            let mut state = self.account_state.write().await;
            state.consecutive_losses += 1;
        } else if record.pnl > 0.0 {
            let mut state = self.account_state.write().await;
            state.consecutive_losses = 0;
        }
    }

    fn calculate_risk_score(
        &self,
        account: &AccountState,
        violations: &[RiskViolation],
    ) -> f64 {
        let mut score = 0.0;

        // Base score from violations
        score += violations.len() as f64 * 0.2;

        // Daily PnL risk
        let pnl_ratio = account.daily_pnl.abs() / self.daily_loss_limit;
        score += pnl_ratio * 0.3;

        // Leverage risk
        let leverage = if account.equity > 0.0 {
            account.margin_used / account.equity
        } else {
            0.0
        };
        score += (leverage / 10.0) * 0.2;

        // Consecutive losses risk
        score += (account.consecutive_losses as f64 / 10.0) * 0.15;

        // Volatility risk
        score += (account.current_volatility / 1.0) * 0.15;

        score.min(1.0)
    }

    async fn calculate_max_allowed_size(
        &self,
        symbol: &str,
        account: &AccountState,
        risk_score: f64,
    ) -> f64 {
        let limits = self.position_limits.read().await;
        
        if let Some(limit) = limits.get(symbol) {
            // Reduce max size based on risk score
            let risk_adjustment = 1.0 - risk_score;
            limit.max_position_size * risk_adjustment
        } else {
            // Default: 1% of equity at 10x leverage
            let default_size = (account.equity * 0.01 * 10.0) / 
                account.positions.get(symbol)
                    .map(|p| p.current_price)
                    .unwrap_or(1.0);
            
            default_size * (1.0 - risk_score)
        }
    }

    pub async fn get_statistics(&self) -> RiskStatistics {
        let checks = *self.total_risk_checks.read().await;
        let violations = *self.total_violations.read().await;
        let account = self.account_state.read().await.clone();
        let peak = *self.peak_equity.read().await;
        let trading_allowed = *self.is_trading_allowed.read().await;

        let current_drawdown = if peak > 0.0 {
            (peak - account.equity) / peak * 100.0
        } else {
            0.0
        };

        RiskStatistics {
            total_risk_checks: checks,
            total_violations: violations,
            violation_rate: if checks > 0 {
                violations as f64 / checks as f64
            } else {
                0.0
            },
            current_equity: account.equity,
            peak_equity: peak,
            current_drawdown_pct: current_drawdown,
            daily_pnl: account.daily_pnl,
            consecutive_losses: account.consecutive_losses,
            trading_allowed,
        }
    }
}

impl Default for AccountState {
    fn default() -> Self {
        Self {
            balance: 10000.0,
            equity: 10000.0,
            margin_used: 0.0,
            positions: HashMap::new(),
            daily_pnl: 0.0,
            consecutive_losses: 0,
            current_volatility: 0.0,
            api_error_rate: 0.0,
            model_disagreement: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RiskStatistics {
    pub total_risk_checks: u64,
    pub total_violations: u64,
    pub violation_rate: f64,
    pub current_equity: f64,
    pub peak_equity: f64,
    pub current_drawdown_pct: f64,
    pub daily_pnl: f64,
    pub consecutive_losses: u32,
    pub trading_allowed: bool,
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("risk_manager=info")
        .with_target(false)
        .with_thread_ids(true)
        .init();

    info!("
    ╔═══════════════════════════════════════════════════════╗
    ║     QUANTUM ALPHA - Risk Manager v0.1.0              ║
    ║     Enterprise Risk Management & Kill Switch         ║
    ╚═══════════════════════════════════════════════════════╝
    ");

    // Configure kill switches
    let kill_switches = vec![
        KillSwitch::DailyLossLimit { threshold: 500.0 },  // $500 daily loss
        KillSwitch::ConsecutiveLosses { max_losses: 5 },
        KillSwitch::VolatilitySpike { threshold: 0.05 },  // 5% volatility
        KillSwitch::ApiErrorRate { threshold: 0.1 },      // 10% error rate
        KillSwitch::DrawdownLimit { max_drawdown_pct: 15.0 },
        KillSwitch::LeverageExcess { max_leverage: 10.0 },
    ];

    // Create risk manager
    let risk_manager = RiskManager::new(
        500.0,   // $500 daily loss limit
        20.0,    // 20% max drawdown
        kill_switches,
    );

    // Set position limits for BTC and ETH
    risk_manager.set_position_limit(
        "BTCUSDT".to_string(),
        PositionLimit {
            max_position_size: 0.1,      // 0.1 BTC
            max_leverage: 10.0,
            max_risk_per_trade_pct: 2.0,
        },
    ).await;

    risk_manager.set_position_limit(
        "ETHUSDT".to_string(),
        PositionLimit {
            max_position_size: 2.0,      // 2 ETH
            max_leverage: 10.0,
            max_risk_per_trade_pct: 2.0,
        },
    ).await;

    info!("✅ Risk Manager initialized");
    info!("🛡️  Kill switches configured: {}", 6);

    // Simulate some risk checks
    info!("\n🧪 Running test risk checks...");

    // Test 1: Normal order
    let result = risk_manager.check_order_risk(
        "BTCUSDT",
        &PositionSide::Long,
        0.01,
        40000.0,
    ).await;

    info!(
        "Test 1 - Normal order: {} (risk score: {:.2})",
        if result.is_approved { "✅ APPROVED" } else { "❌ REJECTED" },
        result.risk_score
    );

    // Test 2: Oversized order
    let result = risk_manager.check_order_risk(
        "BTCUSDT",
        &PositionSide::Long,
        1.0,  // Too large
        40000.0,
    ).await;

    info!(
        "Test 2 - Oversized order: {} (violations: {})",
        if result.is_approved { "✅ APPROVED" } else { "❌ REJECTED" },
        result.violations.len()
    );

    // Test 3: Simulate daily loss limit breach
    let mut test_account = AccountState::default();
    test_account.daily_pnl = -600.0;  // Exceed $500 limit
    risk_manager.update_account_state(test_account).await;

    let result = risk_manager.check_order_risk(
        "BTCUSDT",
        &PositionSide::Long,
        0.01,
        40000.0,
    ).await;

    info!(
        "Test 3 - Daily loss limit breach: {} (violations: {})",
        if result.is_approved { "✅ APPROVED" } else { "❌ REJECTED" },
        result.violations.len()
    );

    // Print statistics
    let stats = risk_manager.get_statistics().await;
    info!("\n📊 Risk Management Statistics:");
    info!("   Total risk checks: {}", stats.total_risk_checks);
    info!("   Total violations: {}", stats.total_violations);
    info!("   Violation rate: {:.2}%", stats.violation_rate * 100.0);
    info!("   Current equity: ${:.2}", stats.current_equity);
    info!("   Daily PnL: ${:.2}", stats.daily_pnl);
    info!("   Trading allowed: {}", stats.trading_allowed);

    info!("\n🎯 Risk Manager ready. Press Ctrl+C to stop.");

    tokio::signal::ctrl_c().await?;

    info!("🛑 Shutting down Risk Manager...");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_manager_creation() {
        let kill_switches = vec![
            KillSwitch::DailyLossLimit { threshold: 500.0 },
        ];

        let rm = RiskManager::new(500.0, 20.0, kill_switches);
        
        let stats = rm.get_statistics().await;
        assert_eq!(stats.total_risk_checks, 0);
        assert!(stats.trading_allowed);
    }

    #[tokio::test]
    async fn test_position_size_check() {
        let rm = RiskManager::new(500.0, 20.0, vec![]);
        
        rm.set_position_limit(
            "BTCUSDT".to_string(),
            PositionLimit {
                max_position_size: 0.1,
                max_leverage: 10.0,
                max_risk_per_trade_pct: 2.0,
            },
        ).await;

        // Normal order should pass
        let result = rm.check_order_risk(
            "BTCUSDT",
            &PositionSide::Long,
            0.05,
            40000.0,
        ).await;

        assert!(result.is_approved);
    }

    #[tokio::test]
    async fn test_daily_loss_limit() {
        let kill_switches = vec![
            KillSwitch::DailyLossLimit { threshold: 500.0 },
        ];

        let rm = RiskManager::new(500.0, 20.0, kill_switches);
        
        // Set account with large daily loss
        let mut account = AccountState::default();
        account.daily_pnl = -600.0;
        rm.update_account_state(account).await;

        let result = rm.check_order_risk(
            "BTCUSDT",
            &PositionSide::Long,
            0.01,
            40000.0,
        ).await;

        assert!(!result.is_approved);
        assert!(result.violations.contains(&RiskViolation::KillSwitchTriggered));
    }
}
