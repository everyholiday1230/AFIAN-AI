#!/bin/bash

# QUANTUM ALPHA - Stop Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}"
echo "🛑 Stopping QUANTUM ALPHA..."
echo -e "${NC}"

# Stop all containers
docker-compose stop

echo -e "${GREEN}✅ All services stopped${NC}"
echo ""
echo "To remove containers and volumes:"
echo "   docker-compose down -v"
