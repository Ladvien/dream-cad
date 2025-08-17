"""Performance alerts and anomaly detection for multi-model system."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for performance alerts."""
    
    # Thresholds
    max_generation_time: float = 300.0  # 5 minutes
    min_success_rate: float = 0.8  # 80%
    max_vram_usage_gb: float = 22.0  # Leave 2GB buffer on 24GB GPU
    max_gpu_temperature: float = 83.0  # Celsius
    max_error_rate: float = 0.2  # 20%
    
    # Performance degradation
    performance_degradation_threshold: float = 1.5  # 50% slower than baseline
    quality_degradation_threshold: float = 0.8  # 20% quality drop
    
    # Resource exhaustion
    min_free_vram_gb: float = 2.0
    min_free_ram_gb: float = 4.0
    min_free_disk_gb: float = 10.0
    
    # Alert settings
    alert_cooldown_minutes: int = 30  # Don't repeat same alert within this time
    enable_email_alerts: bool = False
    email_recipients: List[str] = None
    enable_webhook_alerts: bool = False
    webhook_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Alert:
    """Performance alert."""
    
    alert_id: str
    timestamp: datetime
    severity: str  # "warning", "error", "critical"
    alert_type: str  # "performance", "resource", "quality", "availability"
    model_name: Optional[str]
    message: str
    details: Dict[str, Any]
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PerformanceAlerts:
    """Monitor performance and generate alerts."""
    
    def __init__(
        self,
        config: Optional[AlertConfig] = None,
        storage_dir: Optional[Path] = None,
    ):
        """Initialize performance alerts.
        
        Args:
            config: Alert configuration
            storage_dir: Directory to store alerts
        """
        self.config = config or AlertConfig()
        self.storage_dir = storage_dir or Path("monitoring/alerts")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Load baselines
        self._load_baselines()
    
    def check_metrics(self, model_name: str, metrics: Dict[str, Any]) -> List[Alert]:
        """Check metrics and generate alerts if needed.
        
        Args:
            model_name: Name of the model
            metrics: Current metrics
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check generation time
        if metrics.get("generation_time_seconds", 0) > self.config.max_generation_time:
            alerts.append(self._create_alert(
                severity="warning",
                alert_type="performance",
                model_name=model_name,
                message=f"{model_name} generation time exceeded threshold",
                details={
                    "generation_time": metrics["generation_time_seconds"],
                    "threshold": self.config.max_generation_time,
                },
                suggested_action="Consider reducing model quality settings or switching to faster model",
            ))
        
        # Check VRAM usage
        vram_gb = metrics.get("vram_peak_gb", 0)
        if vram_gb > self.config.max_vram_usage_gb:
            alerts.append(self._create_alert(
                severity="error",
                alert_type="resource",
                model_name=model_name,
                message=f"{model_name} VRAM usage critical",
                details={
                    "vram_gb": vram_gb,
                    "threshold": self.config.max_vram_usage_gb,
                },
                suggested_action="Enable memory optimization or reduce batch size",
            ))
        
        # Check GPU temperature
        gpu_temp = metrics.get("gpu_temperature", 0)
        if gpu_temp > self.config.max_gpu_temperature:
            alerts.append(self._create_alert(
                severity="critical",
                alert_type="resource",
                model_name=model_name,
                message="GPU temperature critical",
                details={
                    "temperature": gpu_temp,
                    "threshold": self.config.max_gpu_temperature,
                },
                suggested_action="Reduce GPU load or improve cooling",
            ))
        
        # Check for errors
        if not metrics.get("success", True):
            error_msg = metrics.get("error_message", "Unknown error")
            alerts.append(self._create_alert(
                severity="error",
                alert_type="availability",
                model_name=model_name,
                message=f"{model_name} generation failed",
                details={"error": error_msg},
                suggested_action="Check model configuration and system resources",
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def check_model_stats(
        self,
        model_name: str,
        stats: Dict[str, Any],
    ) -> List[Alert]:
        """Check model statistics for anomalies.
        
        Args:
            model_name: Name of the model
            stats: Model statistics
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check success rate
        success_rate = stats.get("success_rate", 1.0)
        if success_rate < self.config.min_success_rate:
            alerts.append(self._create_alert(
                severity="warning",
                alert_type="quality",
                model_name=model_name,
                message=f"{model_name} success rate below threshold",
                details={
                    "success_rate": success_rate,
                    "threshold": self.config.min_success_rate,
                    "total_generations": stats.get("total_generations", 0),
                },
                suggested_action="Review recent errors and adjust configuration",
            ))
        
        # Check for performance degradation
        if model_name in self.performance_baselines:
            baseline = self.performance_baselines[model_name]
            
            # Check generation time degradation
            avg_time = stats.get("avg_generation_time", 0)
            baseline_time = baseline.get("avg_generation_time", avg_time)
            
            if baseline_time > 0 and avg_time > baseline_time * self.config.performance_degradation_threshold:
                alerts.append(self._create_alert(
                    severity="warning",
                    alert_type="performance",
                    model_name=model_name,
                    message=f"{model_name} performance degradation detected",
                    details={
                        "current_avg_time": avg_time,
                        "baseline_avg_time": baseline_time,
                        "degradation_factor": avg_time / baseline_time if baseline_time > 0 else 0,
                    },
                    suggested_action="Check system resources and recent changes",
                ))
            
            # Check quality degradation
            avg_quality = stats.get("avg_quality_score", 0)
            baseline_quality = baseline.get("avg_quality_score", avg_quality)
            
            if baseline_quality > 0 and avg_quality < baseline_quality * self.config.quality_degradation_threshold:
                alerts.append(self._create_alert(
                    severity="warning",
                    alert_type="quality",
                    model_name=model_name,
                    message=f"{model_name} quality degradation detected",
                    details={
                        "current_quality": avg_quality,
                        "baseline_quality": baseline_quality,
                        "degradation_factor": avg_quality / baseline_quality if baseline_quality > 0 else 0,
                    },
                    suggested_action="Review model configuration and recent changes",
                ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def check_system_resources(self, resources: Dict[str, Any]) -> List[Alert]:
        """Check system resources and generate alerts.
        
        Args:
            resources: System resource metrics
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check free VRAM
        vram_free = resources.get("vram_free_gb", float('inf'))
        if vram_free < self.config.min_free_vram_gb:
            alerts.append(self._create_alert(
                severity="error",
                alert_type="resource",
                model_name=None,
                message="Low VRAM available",
                details={
                    "vram_free_gb": vram_free,
                    "threshold": self.config.min_free_vram_gb,
                },
                suggested_action="Unload unused models or reduce memory usage",
            ))
        
        # Check free RAM
        ram_free = resources.get("ram_free_gb", float('inf'))
        if ram_free < self.config.min_free_ram_gb:
            alerts.append(self._create_alert(
                severity="warning",
                alert_type="resource",
                model_name=None,
                message="Low system RAM available",
                details={
                    "ram_free_gb": ram_free,
                    "threshold": self.config.min_free_ram_gb,
                },
                suggested_action="Close unnecessary applications",
            ))
        
        # Check disk space
        disk_free = resources.get("disk_free_gb", float('inf'))
        if disk_free < self.config.min_free_disk_gb:
            alerts.append(self._create_alert(
                severity="warning",
                alert_type="resource",
                model_name=None,
                message="Low disk space available",
                details={
                    "disk_free_gb": disk_free,
                    "threshold": self.config.min_free_disk_gb,
                },
                suggested_action="Clean up old outputs or expand storage",
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def update_baseline(self, model_name: str, stats: Dict[str, Any]) -> None:
        """Update performance baseline for a model.
        
        Args:
            model_name: Name of the model
            stats: Current statistics to use as baseline
        """
        self.performance_baselines[model_name] = {
            "avg_generation_time": stats.get("avg_generation_time", 0),
            "avg_quality_score": stats.get("avg_quality_score", 0),
            "success_rate": stats.get("success_rate", 1.0),
            "avg_vram_gb": stats.get("avg_vram_gb", 0),
            "updated": datetime.now().isoformat(),
        }
        
        self._save_baselines()
        logger.info(f"Updated baseline for {model_name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a custom alert handler.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge and clear an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            del self.active_alerts[alert_id]
            
            logger.info(f"Acknowledged alert: {alert_id}")
            return True
        
        return False
    
    def _create_alert(
        self,
        severity: str,
        alert_type: str,
        model_name: Optional[str],
        message: str,
        details: Dict[str, Any],
        suggested_action: Optional[str] = None,
    ) -> Alert:
        """Create a new alert."""
        alert_id = f"{alert_type}_{model_name or 'system'}_{datetime.now().timestamp()}"
        
        return Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            alert_type=alert_type,
            model_name=model_name,
            message=message,
            details=details,
            suggested_action=suggested_action,
        )
    
    def _process_alert(self, alert: Alert) -> None:
        """Process an alert."""
        # Check cooldown
        alert_key = f"{alert.alert_type}_{alert.model_name}_{alert.message}"
        
        if alert_key in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_key]
            if time_since_last < timedelta(minutes=self.config.alert_cooldown_minutes):
                logger.debug(f"Alert in cooldown: {alert_key}")
                return
        
        # Record alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = datetime.now()
        
        # Save alert
        self._save_alert(alert)
        
        # Log alert
        log_func = {
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical,
        }.get(alert.severity, logger.info)
        
        log_func(f"Alert: {alert.message} - {alert.details}")
        
        # Send notifications
        if self.config.enable_email_alerts:
            self._send_email_alert(alert)
        
        if self.config.enable_webhook_alerts:
            self._send_webhook_alert(alert)
        
        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _send_email_alert(self, alert: Alert) -> None:
        """Send email alert."""
        if not self.config.email_recipients:
            return
        
        try:
            # Create email
            subject = f"[{alert.severity.upper()}] Dream-CAD Alert: {alert.message}"
            
            body = f"""
Alert Details:
--------------
Time: {alert.timestamp}
Severity: {alert.severity}
Type: {alert.alert_type}
Model: {alert.model_name or 'System'}

Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

Suggested Action:
{alert.suggested_action or 'No specific action recommended'}
            """
            
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = "dream-cad@localhost"
            msg["To"] = ", ".join(self.config.email_recipients)
            
            # Send email (simplified - would need proper SMTP config)
            logger.info(f"Email alert would be sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert) -> None:
        """Send webhook alert."""
        if not self.config.webhook_url:
            return
        
        try:
            # Would send webhook here
            logger.info(f"Webhook alert would be sent: {alert.message}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _save_alert(self, alert: Alert) -> None:
        """Save alert to disk."""
        try:
            file_path = self.storage_dir / "alerts.jsonl"
            with file_path.open("a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
    
    def _save_baselines(self) -> None:
        """Save performance baselines."""
        try:
            file_path = self.storage_dir / "baselines.json"
            with file_path.open("w") as f:
                json.dump(self.performance_baselines, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")
    
    def _load_baselines(self) -> None:
        """Load performance baselines."""
        try:
            file_path = self.storage_dir / "baselines.json"
            if file_path.exists():
                with file_path.open() as f:
                    self.performance_baselines = json.load(f)
                logger.info(f"Loaded baselines for {len(self.performance_baselines)} models")
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")