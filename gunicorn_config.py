# Gunicorn configuration file for production
import multiprocessing
import os

# Get the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, 'logs')

# Create logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Server socket
bind = "127.0.0.1:8001"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Logging
accesslog = os.path.join(log_dir, 'access.log')
errorlog = os.path.join(log_dir, 'error.log')
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'ai-desert'

# Server mechanics
daemon = False
pidfile = os.path.join(base_dir, 'ai-desert.pid')
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed later)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'
