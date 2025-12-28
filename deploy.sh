#!/bin/bash
# Production Deployment Script for AI Desert Recipe Book
# Run this script on your Linux server

echo "=== AI Desert Production Deployment ==="
echo ""

# Configuration
APP_NAME="ai-desert"
APP_DIR="$(pwd)"
VENV_DIR="$APP_DIR/venv"
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
cd "$APP_DIR"

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
echo -e "${YELLOW}Do you have a domain name?${NC}"
echo "1) No, I will access via IP address (e.g., http://192.168.1.100/recipes)"
echo "2) Yes, I have a domain name"
read -p "Enter choice (1 or 2): " domain_choice

if [ "$domain_choice" = "2" ]; then
    read -p "Enter your domain name (e.g., yourdomain.com): " domain_name
    sudo sed -i "s|server_name _;|server_name $domain_name;|g" $APP_DIR/nginx-ai-desert.conf
    sudo sed -i "s|listen 80 default_server;|listen 80;|g" $APP_DIR/nginx-ai-desert.conf
fi

# Copy nginx config
sudo cp $APP_DIR/nginx-ai-desert.conf /etc/nginx/sites-available/$APP_NAME
sudo ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/

# Remove default nginx site if it exists
if [ -f "/etc/nginx/sites-enabled/default" ]; then
    sudo rm /etc/nginx/sites-enabled/default
fi

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

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

if [ "$domain_choice" = "2" ]; then
    echo -e "${GREEN}Access your sites:${NC}"
    echo "  Site 1 (port 8000): http://$domain_name"
    echo "  AI Desert:          http://$domain_name/recipes"
    echo ""
    echo -e "${YELLOW}To enable SSL:${NC} sudo certbot --nginx -d $domain_name"
    echo ""
    echo -e "${YELLOW}Make sure to:${NC}"
    echo "  1. Point your domain DNS to this server IP: $SERVER_IP"
    echo "  2. Open port 80 (and 443 for HTTPS) in your firewall"
else
    echo -e "${GREEN}Access your sites via IP:${NC}"
    echo "  Site 1 (port 8000): http://$SERVER_IP"
    echo "  AI Desert:          http://$SERVER_IP/recipes"
    echo ""
    echo -e "${YELLOW}Make sure to:${NC}"
    echo "  1. Open port 80 in your firewall: sudo ufw allow 80/tcp"
fi

echo "  2. Check that .env file has all required API keys"
echo ""
echo -e "${GREEN}When you get a domain name later:${NC}"
echo "  1. Edit /etc/nginx/sites-available/$APP_NAME"
echo "  2. Change 'server_name _;' to 'server_name yourdomain.com;'"
echo "  3. Restart nginx: sudo systemctl restart nginx"
