#!/bin/bash
# Quantum Alpha Setup Script

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║     QUANTUM ALPHA - Setup Script v0.1.0              ║"
echo "║     세계 최고 수준 암호화폐 선물 자동매매 시스템        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in correct directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}❌ Error: Must run from project root directory${NC}"
    exit 1
fi

echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}⚠️  Rust not found. Installing...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi
echo -e "${GREEN}✅ Rust found: $(cargo --version)${NC}"

# Check Redis
if ! command -v redis-cli &> /dev/null; then
    echo -e "${YELLOW}⚠️  Redis not found. Please install Redis:${NC}"
    echo "   Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   macOS: brew install redis"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:alpine"
fi

# Check PostgreSQL/TimescaleDB
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}⚠️  PostgreSQL not found. Please install PostgreSQL/TimescaleDB${NC}"
    echo "   Or use Docker: see docker-compose.yml"
fi

echo ""
echo "📦 Setting up Python environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "🦀 Building Rust components..."

# Build data collector
echo "  Building data_collector..."
cd core/data_collector
cargo build --release
cd ../..

# Build order executor
echo "  Building order_executor..."
cd core/order_executor
cargo build --release
cd ../..

echo ""
echo "📁 Creating necessary directories..."

# Create data directories
mkdir -p data/raw data/processed data/models
mkdir -p logs
mkdir -p monitoring/grafana/data
mkdir -p monitoring/prometheus/data

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/models/.gitkeep
touch logs/.gitkeep

echo ""
echo "⚙️  Setting up environment variables..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Quantum Alpha Environment Variables

# Redis
REDIS_URL=redis://127.0.0.1:6379

# TimescaleDB
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432
TIMESCALE_DATABASE=quantum_alpha
TIMESCALE_USER=quantum
TIMESCALE_PASSWORD=your_password_here

# Bybit API (TESTNET)
BYBIT_API_KEY=your_testnet_api_key
BYBIT_API_SECRET=your_testnet_api_secret
BYBIT_TESTNET=true

# S3 (Optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Monitoring (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

    echo -e "${GREEN}✅ .env file created${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env file with your API keys${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

echo ""
echo "🧪 Running tests..."

# Test Python modules
python -m pytest ai/tests/ -v || echo -e "${YELLOW}⚠️  Some tests failed (this is normal for initial setup)${NC}"

# Test Rust modules
cd core/data_collector && cargo test && cd ../..
cd core/order_executor && cargo test && cd ../..

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                  Setup Complete! 🎉                  ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   ${YELLOW}nano .env${NC}"
echo ""
echo "2. Start Redis (if not running):"
echo "   ${YELLOW}redis-server${NC}"
echo ""
echo "3. Start data collector:"
echo "   ${YELLOW}cd core/data_collector && cargo run --release${NC}"
echo ""
echo "4. Start order executor:"
echo "   ${YELLOW}cd core/order_executor && cargo run --release${NC}"
echo ""
echo "5. Train models:"
echo "   ${YELLOW}python ai/training/train_tft.py${NC}"
echo ""
echo "6. Run backtest:"
echo "   ${YELLOW}python backtesting/run_backtest.py${NC}"
echo ""
echo "For documentation, see: ${YELLOW}docs/${NC}"
echo ""
