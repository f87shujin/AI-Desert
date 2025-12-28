#!/bin/bash
# Production Deployment Script for AI Desert Recipe Book
# Run this script on your Linux server

echo "=== AI Desert Production Deployment ==="
echo ""

# Configuration
APP_NAME="ai-desert"
APP_DIR="/home/$(whoami)/AI-Desert"
VENV_DIR="$APP_DIR/.venv"
LOG_DIR="/var/log/$APP_NAME"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Update system
echo -e "${GREEN}Step 1: Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Step 2: Install dependencies
echo -e "${GREEN}Step 2: Installing dependencies...${NC}"
sudo apt install -y python3 python3-pip python3-venv nginx git

# Step 3: Clone or update repository
echo -e "${GREEN}Step 3: Setting up application directory...${NC}"
if [ ! -d "$APP_DIR" ]; then
    echo "Creating directory: $APP_DIR"
    mkdir -p $APP_DIR
fi

cd $APP_DIR

# Step 4: Create virtual environment
echo -e "${GREEN}Step 4: Creating Python virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Step 5: Install Python packages
echo -e "${GREEN}Step 5: Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Step 6: Create log directory
echo -e "${GREEN}Step 6: Creating log directory...${NC}"
sudo mkdir -p $LOG_DIR
sudo chown -R $(whoami):$(whoami) $LOG_DIR

# Step 7: Set up environment variables
echo -e "${GREEN}Step 7: Checking .env file...${NC}"
if [ ! -f "$APP_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "Please create .env file with your configuration:"
    echo "  - DEEPSEEK_API_KEY"
    echo "  - MONGODB_URI"
    echo "  - GOOGLE_API_KEY_1, GOOGLE_API_KEY_2"
    echo "  - GOOGLE_SEARCH_ENGINE_ID"
    echo "  - SECRET_KEY"
    echo ""
    read -p "Press enter when .env file is ready..."
fi

# Step 8: Update systemd service file
echo -e "${GREEN}Step 8: Setting up systemd service...${NC}"
# Update paths in service file
sudo sed -i "s|/home/yourusername|/home/$(whoami)|g" $APP_DIR/ai-desert.service

# Copy service file
sudo cp $APP_DIR/ai-desert.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Step 9: Set up Nginx
echo -e "${GREEN}Step 9: Configuring Nginx...${NC}"
# Update paths in nginx config
sudo sed -i "s|/home/yourusername|/home/$(whoami)|g" $APP_DIR/nginx-ai-desert.conf

echo ""
echo -e "${YELLOW}Choose how to access your site:${NC}"
echo "1) Subdomain (e.g., recipes.yourdomain.com) - Recommended"
echo "2) Path (e.g., yourdomain.com/recipes)"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    read -p "Enter your subdomain (e.g., recipes.yourdomain.com): " subdomain
    sudo sed -i "s|recipes.yourdomain.com|$subdomain|g" $APP_DIR/nginx-ai-desert.conf
fi

# Copy nginx config
sudo cp $APP_DIR/nginx-ai-desert.conf /etc/nginx/sites-available/$APP_NAME
sudo ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/

# Test nginx config
sudo nginx -t

# Step 10: Start services
echo -e "${GREEN}Step 10: Starting services...${NC}"
sudo systemctl enable $APP_NAME
sudo systemctl start $APP_NAME
sudo systemctl restart nginx

# Step 11: Check status
echo ""
echo -e "${GREEN}=== Deployment Complete! ===${NC}"
echo ""
echo "Service Status:"
sudo systemctl status $APP_NAME --no-pager | head -n 10

echo ""
echo -e "${GREEN}Useful Commands:${NC}"
echo "  View logs:        sudo journalctl -u $APP_NAME -f"
echo "  Restart app:      sudo systemctl restart $APP_NAME"
echo "  Stop app:         sudo systemctl stop $APP_NAME"
echo "  Check status:     sudo systemctl status $APP_NAME"
echo "  Restart Nginx:    sudo systemctl restart nginx"
echo ""

if [ "$choice" = "1" ]; then
    echo -e "${GREEN}Access your app at: http://$subdomain${NC}"
    echo -e "${YELLOW}To enable SSL:${NC} sudo certbot --nginx -d $subdomain"
else
    echo -e "${GREEN}Access your app at: http://yourdomain.com/recipes${NC}"
fi

echo ""
echo -e "${YELLOW}Note: Make sure to:${NC}"
echo "  1. Configure your domain DNS to point to this server"
echo "  2. Open port 80 (and 443 for HTTPS) in your firewall"
echo "  3. Check that .env file has all required API keys"
