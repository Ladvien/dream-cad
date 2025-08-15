# MVDream Production Setup Guide

This guide covers setting up MVDream for production use with monitoring, logging, and automatic recovery.

## Overview

The production setup provides:
- Comprehensive logging with rotation
- Real-time GPU and system monitoring
- Automatic checkpoint saving and recovery
- Queue-based batch processing
- Resource alerts and notifications
- Background operation support

## Quick Start

### 1. Start Production Manager

Start the full production system:
```bash
poetry run python scripts/production_monitor.py start
```

This starts:
- GPU monitoring (every 30 seconds)
- Queue processor
- Checkpoint manager
- Alert system

### 2. Monitor Only Mode

For monitoring without processing:
```bash
poetry run python scripts/production_monitor.py monitor --interval 30
```

### 3. Check System Status

View current system status:
```bash
poetry run python scripts/production_monitor.py status
```

Output shows:
- GPU metrics (temperature, VRAM, utilization)
- System metrics (CPU, RAM, disk)
- Queue status (pending, running, completed jobs)

## Logging Configuration

### Log Files

All logs are stored in `/mnt/datadrive_m2/dream-cad/logs/`:

- `mvdream.production.log` - Main production log
- `mvdream.alerts.log` - Resource alerts
- `gpu_metrics.jsonl` - GPU metrics (JSON lines format)
- `generation_metrics.json` - Generation performance metrics

### Log Rotation

Logs rotate automatically:
- Max size: 10MB per file
- Backup count: 10 files
- Total max: 100MB per log type

### Log Levels

Configure in scripts:
```python
logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Viewing Logs

Real-time monitoring:
```bash
tail -f logs/mvdream.production.log
```

Search logs:
```bash
grep "ERROR" logs/mvdream.production.log
```

Parse GPU metrics:
```bash
jq '.temperature_c' logs/gpu_metrics.jsonl
```

## GPU Monitoring

### Automatic Monitoring

GPU metrics logged every 30 seconds:
- Temperature (°C)
- VRAM usage (GB and %)
- Utilization (%)
- Power draw (W)

### Manual Monitoring

Real-time GPU stats:
```bash
poetry run poe monitor
```

Or use nvidia-smi:
```bash
watch -n 1 nvidia-smi
```

### Resource Alerts

Automatic alerts trigger when:
- VRAM usage > 90%
- GPU temperature > 83°C
- RAM usage > 90%
- Disk usage > 90%

Alerts are logged to:
- `logs/mvdream.alerts.log`
- Main production log
- Console output (if attached)

## Checkpoint System

### Automatic Checkpointing

Checkpoints save every 1000 steps:
```python
checkpoint_interval = 1000  # Configurable
```

Location: `/mnt/datadrive_m2/dream-cad/checkpoints/`

### Manual Checkpoint Recovery

Recover from latest checkpoint:
```bash
poetry run python scripts/production_monitor.py recover job_20250815_120000_0
```

### Checkpoint Management

Automatic cleanup keeps last 3 checkpoints per job.

Manual cleanup:
```bash
find checkpoints/ -name "*.ckpt" -mtime +7 -delete  # Remove >7 days old
```

## Queue System

### Adding Jobs

Add single job:
```bash
poetry run python scripts/production_monitor.py queue add "a ceramic coffee mug"
```

With custom config:
```bash
poetry run python scripts/production_monitor.py queue add "prompt" --config custom.yaml
```

### Viewing Queue

List all jobs:
```bash
poetry run python scripts/production_monitor.py queue list
```

### Queue File

Queue persisted in: `/mnt/datadrive_m2/dream-cad/generation_queue.json`

Format:
```json
{
  "jobs": [
    {
      "id": "job_20250815_120000_0",
      "prompt": "a ceramic coffee mug",
      "status": "pending",
      "created_at": "2025-08-15T12:00:00",
      "config": {}
    }
  ]
}
```

### Batch Processing

Add multiple jobs via script:
```python
from scripts.production_monitor import GenerationQueue

queue = GenerationQueue()
prompts = ["prompt1", "prompt2", "prompt3"]
for prompt in prompts:
    queue.add_job(prompt, {})
```

## Poethepoet Tasks

### Available Tasks

```bash
poetry run poe monitor     # Real-time GPU monitoring
poetry run poe queue-add   # Add job to queue
poetry run poe queue-list  # List queue jobs
poetry run poe prod-start  # Start production manager
poetry run poe prod-status # Show system status
```

### Adding Custom Tasks

Edit `pyproject.toml`:
```toml
[tool.poe.tasks]
monitor = "python scripts/production_monitor.py monitor"
prod-start = "python scripts/production_monitor.py start"
```

## Systemd Service (Optional)

### Create Service File

Create `/etc/systemd/system/mvdream.service`:
```ini
[Unit]
Description=MVDream Production Manager
After=network.target

[Service]
Type=simple
User=youruser
Group=yourgroup
WorkingDirectory=/mnt/datadrive_m2/dream-cad
Environment="PATH=/mnt/datadrive_m2/dream-cad/.venv/bin:/usr/bin"
ExecStart=/mnt/datadrive_m2/dream-cad/.venv/bin/python scripts/production_monitor.py start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable mvdream
sudo systemctl start mvdream
```

### Check Status

```bash
sudo systemctl status mvdream
sudo journalctl -u mvdream -f  # View logs
```

## Production Best Practices

### 1. Pre-Production Checklist

- [ ] Disk space > 100GB free
- [ ] GPU drivers updated
- [ ] System packages updated
- [ ] Python dependencies installed
- [ ] Models downloaded and verified
- [ ] Logs directory has write permissions
- [ ] Checkpoint directory has write permissions

### 2. Resource Management

**Memory Optimization:**
```yaml
# configs/mvdream-sd21.yaml
generation:
  batch_size: 1  # Reduce for production stability
  enable_cpu_offload: true  # If hitting VRAM limits
```

**Temperature Management:**
```bash
# Set GPU power limit
sudo nvidia-smi -pl 300  # Reduce to 300W

# Set fan curve
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"
sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=80"
```

### 3. Monitoring Strategy

**Key Metrics to Track:**
- Generation time per job
- Success/failure rate
- Average GPU temperature
- Peak VRAM usage
- Queue throughput

**Alert Thresholds:**
```python
ALERT_THRESHOLDS = {
    "gpu_temp_c": 83,
    "vram_percent": 90,
    "ram_percent": 90,
    "disk_percent": 90,
    "queue_size": 100,
}
```

### 4. Backup and Recovery

**Regular Backups:**
```bash
# Backup script (run daily)
#!/bin/bash
BACKUP_DIR="/backup/mvdream/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
cp -r checkpoints/ $BACKUP_DIR/
cp -r logs/ $BACKUP_DIR/
cp generation_queue.json $BACKUP_DIR/
```

**Recovery Procedure:**
1. Stop production manager
2. Restore from backup
3. Check latest checkpoints
4. Resume queue processing

### 5. Scaling Considerations

**Multiple GPU Support:**
```python
# Future enhancement
gpu_ids = [0, 1]  # Use multiple GPUs
parallel_jobs = len(gpu_ids)
```

**Distributed Queue:**
- Consider Redis for distributed queue
- Use Celery for task distribution
- Implement load balancing

## Troubleshooting

### Common Issues

**1. High Memory Usage**
```bash
# Clear GPU cache
poetry run python -c "import torch; torch.cuda.empty_cache()"

# Restart production manager
sudo systemctl restart mvdream
```

**2. Stuck Jobs**
```bash
# Manually update job status
poetry run python -c "
from scripts.production_monitor import GenerationQueue
q = GenerationQueue()
q.update_job_status('job_id', 'failed', error_message='Manual intervention')
"
```

**3. Checkpoint Corruption**
```bash
# Remove corrupted checkpoint
rm checkpoints/job_*_corrupted.ckpt

# Resume from earlier checkpoint
poetry run python scripts/production_monitor.py recover job_id
```

### Debug Mode

Enable debug logging:
```python
# In production_monitor.py
logger.setLevel(logging.DEBUG)
```

View debug logs:
```bash
grep "DEBUG" logs/mvdream.production.log | tail -100
```

## Performance Metrics

### Tracking Metrics

Generation metrics saved to `logs/generation_metrics.json`:
```json
{
  "total_jobs": 100,
  "successful": 95,
  "failed": 5,
  "average_time_minutes": 45,
  "average_vram_gb": 16,
  "peak_temperature_c": 78
}
```

### Analyzing Performance

```python
import json
import statistics

# Load metrics
with open("logs/gpu_metrics.jsonl") as f:
    metrics = [json.loads(line) for line in f]

# Analyze
temps = [m["temperature_c"] for m in metrics]
print(f"Average temp: {statistics.mean(temps):.1f}°C")
print(f"Max temp: {max(temps)}°C")
```

## Security Considerations

### File Permissions

```bash
# Secure logs
chmod 600 logs/*.log
chmod 700 logs/

# Secure checkpoints
chmod 600 checkpoints/*.ckpt
chmod 700 checkpoints/
```

### Access Control

- Run as non-root user
- Use systemd security features:
  ```ini
  PrivateTmp=true
  ProtectSystem=strict
  ProtectHome=true
  ```

### Monitoring Access

- Use SSH for remote monitoring
- Set up VPN for external access
- Implement API authentication for web interface

## Integration with CI/CD

### GitHub Actions

```yaml
# .github/workflows/production-deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v2
      - name: Restart production
        run: |
          sudo systemctl stop mvdream
          poetry install
          sudo systemctl start mvdream
```

### Health Checks

```python
# healthcheck.py
import requests

def check_health():
    # Check if service is running
    response = requests.get("http://localhost:7860/health")
    assert response.status_code == 200
    
    # Check GPU
    metrics = get_gpu_metrics()
    assert metrics.temperature_c < 85
    
    return True
```

## Maintenance

### Daily Tasks
- Check logs for errors
- Monitor disk space
- Review alert log

### Weekly Tasks
- Clean old checkpoints
- Archive completed jobs
- Review performance metrics

### Monthly Tasks
- Update dependencies
- Clean old logs
- Performance analysis
- Capacity planning

## Summary

The production setup provides:
1. **Reliability** - Automatic recovery from failures
2. **Observability** - Comprehensive logging and monitoring
3. **Scalability** - Queue-based processing
4. **Maintainability** - Clear structure and documentation

For additional help, see:
- [Troubleshooting Guide](troubleshooting.md)
- [Performance Tuning](performance_tuning.md)
- [System Requirements](system-specs.md)