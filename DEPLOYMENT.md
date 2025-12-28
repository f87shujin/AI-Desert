# AI Desert - Production Deployment Guide

## Quick Deployment

### 1. Transfer files to your Linux server
```bash
# On your local machine (Windows)
scp -r AI-Desert yourusername@yourserver:/home/yourusername/

# Or use git (recommended)
ssh yourusername@yourserver
cd /home/yourusername
git clone https://github.com/yourusername/AI-Desert.git
cd AI-Desert
```

### 2. Run the deployment script
```bash
chmod +x deploy.sh
./deploy.sh
```

The script will automatically:
- Install all dependencies
- Set up Python virtual environment
- Configure systemd service
- Set up Nginx reverse proxy
- Start the application

---

## Manual Deployment Steps

If you prefer manual setup:

### 1. Install dependencies
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx
```

### 2. Set up Python environment
```bash
cd /home/yourusername/AI-Desert
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
nano .env
# Add all your API keys and configuration
```

### 4. Create log directory
```bash
sudo mkdir -p /var/log/ai-desert
sudo chown -R $USER:$USER /var/log/ai-desert
```

### 5. Set up systemd service
```bash
# Edit service file to update paths
nano ai-desert.service
# Replace /home/yourusername with your actual path

# Copy and enable service
sudo cp ai-desert.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-desert
sudo systemctl start ai-desert
```

### 6. Configure Nginx
```bash
# Edit nginx config
nano nginx-ai-desert.conf
# Update domain names and paths

# Copy config
sudo cp nginx-ai-desert.conf /etc/nginx/sites-available/ai-desert
sudo ln -s /etc/nginx/sites-available/ai-desert /etc/nginx/sites-enabled/

# Test and restart
sudo nginx -t
sudo systemctl restart nginx
```

---

## Accessing Multiple Sites

You have **3 options** to run multiple sites on the same server:

### Option 1: Different Subdomains (RECOMMENDED) ✅
```
Site 1: site1.yourdomain.com  → Port 8000
Site 2: recipes.yourdomain.com → Port 8001
```

**Setup:**
- Each site gets its own subdomain
- Create separate nginx config for each
- Point DNS A records for each subdomain to your server IP

**Nginx config example:**
```nginx
# Site 1
server {
    listen 80;
    server_name site1.yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}

# Site 2 (AI Desert)
server {
    listen 80;
    server_name recipes.yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8001;
    }
}
```

### Option 2: Different Paths
```
yourdomain.com/         → Port 8000 (main site)
yourdomain.com/recipes  → Port 8001 (AI Desert)
```

**Nginx config example:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
    }
    
    location /recipes/ {
        proxy_pass http://127.0.0.1:8001/;
    }
}
```

### Option 3: Different Domains
```
domain1.com → Port 8000
domain2.com → Port 8001
```

Same as Option 1, but with completely different domains.

---

## Management Commands

```bash
# View logs
sudo journalctl -u ai-desert -f

# Restart application
sudo systemctl restart ai-desert

# Check status
sudo systemctl status ai-desert

# Stop application
sudo systemctl stop ai-desert

# Restart Nginx
sudo systemctl restart nginx

# Test Nginx config
sudo nginx -t

# View Nginx error logs
sudo tail -f /var/log/nginx/error.log

# View app logs
sudo tail -f /var/log/ai-desert/error.log
```

---

## SSL/HTTPS Setup (Recommended)

Install Certbot and get free SSL certificate:

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate (subdomain method)
sudo certbot --nginx -d recipes.yourdomain.com

# Get certificate for multiple domains
sudo certbot --nginx -d yourdomain.com -d recipes.yourdomain.com

# Auto-renewal is set up automatically
# Test renewal:
sudo certbot renew --dry-run
```

---

## Firewall Setup

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp  # SSH (make sure this is open!)

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

---

## Troubleshooting

### App won't start
```bash
# Check logs
sudo journalctl -u ai-desert -n 50

# Check if port is in use
sudo netstat -tlnp | grep 8001

# Test app manually
cd /home/yourusername/AI-Desert
source .venv/bin/activate
python app.py
```

### Nginx errors
```bash
# Check nginx error log
sudo tail -f /var/log/nginx/error.log

# Test configuration
sudo nginx -t

# Check if nginx is running
sudo systemctl status nginx
```

### Permission errors
```bash
# Fix ownership
sudo chown -R $USER:$USER /home/yourusername/AI-Desert
sudo chown -R $USER:$USER /var/log/ai-desert

# Fix permissions
chmod -R 755 /home/yourusername/AI-Desert
```

### MongoDB connection issues
- Check MONGODB_URI in .env file
- Whitelist server IP in MongoDB Atlas
- Test connection: `ping cluster0.fqatder.mongodb.net`

---

## Performance Optimization

Edit `gunicorn_config.py` to adjust workers:
```python
# More workers = handle more concurrent requests
workers = 4  # Adjust based on your CPU cores
```

---

## Updating the Application

```bash
cd /home/yourusername/AI-Desert

# Pull latest changes (if using git)
git pull

# Activate venv
source .venv/bin/activate

# Install any new dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart ai-desert
```

---

## Monitoring

### Check if app is running
```bash
curl http://localhost:8001
```

### Monitor resource usage
```bash
# CPU and memory
htop

# Disk space
df -h

# App processes
ps aux | grep gunicorn
```

---

## Security Checklist

- ✅ Use HTTPS (SSL certificate)
- ✅ Keep `.env` file secure (never commit to git)
- ✅ Enable firewall (ufw)
- ✅ Regular system updates: `sudo apt update && sudo apt upgrade`
- ✅ Use strong SECRET_KEY in .env
- ✅ Whitelist MongoDB IP addresses
- ✅ Set up fail2ban for SSH protection
- ✅ Disable root SSH login
- ✅ Use SSH keys instead of passwords
