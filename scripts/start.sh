#!/bin/bash

# QUANTUM ALPHA - Startup Script
# 세계 최고 수준 암호화폐 선물 자동매매 시스템

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                   QUANTUM ALPHA v0.1.0                        ║"
echo "║        세계 최고 수준 암호화폐 선물 자동매매 시스템              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${RED}❌ Please edit .env file with your API keys before running!${NC}"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

echo -e "${GREEN}🔧 Environment Configuration${NC}"
echo "   Mode: ${SYSTEM_MODE:-paper_trading}"
echo "   Log Level: ${LOG_LEVEL:-INFO}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is installed${NC}"

# Create necessary directories
echo -e "${GREEN}📁 Creating directories...${NC}"
mkdir -p logs models data/historical data/realtime monitoring/grafana/dashboards

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker daemon is not running!${NC}"
    exit 1
fi

# Build and start services
echo -e "${GREEN}🐳 Starting Docker services...${NC}"
docker-compose up -d --build

echo ""
echo -e "${GREEN}⏳ Waiting for services to be ready...${NC}"

# Wait for TimescaleDB
echo -n "   TimescaleDB: "
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker-compose exec -T timescaledb pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ready${NC}"
        break
    fi
    echo -n "."
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}❌ Failed${NC}"
    exit 1
fi

# Wait for Redis
echo -n "   Redis: "
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ready${NC}"
        break
    fi
    echo -n "."
    sleep 1
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}❌ Failed${NC}"
    exit 1
fi

# Wait for other services
sleep 5

echo ""
echo -e "${GREEN}🎯 Services Status${NC}"
docker-compose ps

echo ""
echo -e "${GREEN}📊 Access Points${NC}"
echo "   Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo "   TimescaleDB: localhost:5432"
echo "   Redis: localhost:6379"

echo ""
echo -e "${GREEN}📝 Viewing Logs${NC}"
echo "   All services: docker-compose logs -f"
echo "   Main system: docker-compose logs -f quantum_alpha"
echo "   Data collector: docker-compose logs -f data_collector"
echo "   Order executor: docker-compose logs -f order_executor"
echo "   Risk manager: docker-compose logs -f risk_manager"

echo ""
echo -e "${GREEN}🛑 Stopping System${NC}"
echo "   Stop: ./scripts/stop.sh"
echo "   Stop & Clean: docker-compose down -v"

echo ""
echo -e "${GREEN}✅ QUANTUM ALPHA is now running!${NC}"
echo ""

# Option to follow main logs
read -p "Follow main system logs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose logs -f quantum_alpha
fi
