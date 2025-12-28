# Gunicorn configuration file for production
import multiprocessing

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
accesslog = '/var/log/ai-desert/access.log'
errorlog = '/var/log/ai-desert/error.log'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'ai-desert'

# Server mechanics
daemon = False
pidfile = '/var/run/ai-desert.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed later)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'
