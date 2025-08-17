"""Resource forecasting for multi-model 3D generation system."""

import json
import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
# numpy is optional for basic statistics
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Resource forecast result."""
    
    forecast_time: datetime
    time_horizon_minutes: int
    
    # VRAM forecasts
    predicted_vram_gb: float
    vram_confidence_interval: Tuple[float, float]  # (lower, upper)
    vram_peak_probability: float  # Probability of hitting peak
    
    # GPU utilization forecasts
    predicted_gpu_utilization: float
    gpu_overload_probability: float
    
    # Queue-based predictions
    estimated_queue_completion_time: float  # minutes
    concurrent_models_forecast: int
    bottleneck_models: List[str]
    
    # Capacity predictions
    available_capacity_percentage: float
    recommended_batch_size: int
    can_accept_new_jobs: bool
    
    # Risk assessment
    risk_level: str  # "low", "medium", "high"
    risk_factors: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["forecast_time"] = self.forecast_time.isoformat()
        return data


class ResourceForecaster:
    """Forecast resource usage based on queue and historical patterns."""
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        history_window_hours: int = 168,  # 1 week
    ):
        """Initialize resource forecaster.
        
        Args:
            storage_dir: Directory to store forecasts
            history_window_hours: Hours of history to use for forecasting
        """
        self.storage_dir = storage_dir or Path("monitoring/forecasts")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_window = timedelta(hours=history_window_hours)
        
        # Model profiles (learned from history)
        self.model_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Time series data
        self.resource_history: List[Dict[str, Any]] = []
        
        # Load historical data
        self._load_history()
        
    def forecast(
        self,
        queue_jobs: List[Dict[str, Any]],
        current_resources: Dict[str, float],
        time_horizon_minutes: int = 60,
    ) -> ForecastResult:
        """Generate resource usage forecast.
        
        Args:
            queue_jobs: List of pending jobs in queue
            current_resources: Current resource usage
            time_horizon_minutes: Forecast horizon in minutes
            
        Returns:
            Resource forecast result
        """
        # Analyze queue
        queue_analysis = self._analyze_queue(queue_jobs)
        
        # Predict resource usage
        vram_forecast = self._forecast_vram(
            queue_analysis,
            current_resources.get("vram_gb", 0),
            time_horizon_minutes,
        )
        
        gpu_forecast = self._forecast_gpu_utilization(
            queue_analysis,
            current_resources.get("gpu_utilization", 0),
            time_horizon_minutes,
        )
        
        # Estimate completion times
        completion_time = self._estimate_completion_time(queue_jobs)
        
        # Assess capacity
        capacity = self._assess_capacity(
            vram_forecast["predicted"],
            gpu_forecast["predicted"],
            queue_analysis["concurrent_models"],
        )
        
        # Risk assessment
        risk = self._assess_risk(
            vram_forecast,
            gpu_forecast,
            capacity,
            queue_analysis,
        )
        
        # Create forecast result
        return ForecastResult(
            forecast_time=datetime.now(),
            time_horizon_minutes=time_horizon_minutes,
            predicted_vram_gb=vram_forecast["predicted"],
            vram_confidence_interval=vram_forecast["confidence_interval"],
            vram_peak_probability=vram_forecast["peak_probability"],
            predicted_gpu_utilization=gpu_forecast["predicted"],
            gpu_overload_probability=gpu_forecast["overload_probability"],
            estimated_queue_completion_time=completion_time,
            concurrent_models_forecast=queue_analysis["concurrent_models"],
            bottleneck_models=queue_analysis["bottleneck_models"],
            available_capacity_percentage=capacity["available_percentage"],
            recommended_batch_size=capacity["recommended_batch_size"],
            can_accept_new_jobs=capacity["can_accept_jobs"],
            risk_level=risk["level"],
            risk_factors=risk["factors"],
            recommendations=risk["recommendations"],
        )
    
    def update_model_profile(
        self,
        model_name: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Update model resource profile based on observed metrics.
        
        Args:
            model_name: Name of the model
            metrics: Observed metrics for the model
        """
        if model_name not in self.model_profiles:
            self.model_profiles[model_name] = {
                "vram_gb_samples": [],
                "generation_time_samples": [],
                "gpu_utilization_samples": [],
                "success_rate": 1.0,
            }
        
        profile = self.model_profiles[model_name]
        
        # Update samples (keep last 100)
        if "vram_peak_gb" in metrics:
            profile["vram_gb_samples"].append(metrics["vram_peak_gb"])
            profile["vram_gb_samples"] = profile["vram_gb_samples"][-100:]
        
        if "generation_time_seconds" in metrics:
            profile["generation_time_samples"].append(metrics["generation_time_seconds"])
            profile["generation_time_samples"] = profile["generation_time_samples"][-100:]
        
        if "gpu_utilization" in metrics:
            profile["gpu_utilization_samples"].append(metrics["gpu_utilization"])
            profile["gpu_utilization_samples"] = profile["gpu_utilization_samples"][-100:]
        
        # Update success rate
        if "success" in metrics:
            # Exponential moving average
            alpha = 0.1
            profile["success_rate"] = (
                alpha * (1.0 if metrics["success"] else 0.0) +
                (1 - alpha) * profile["success_rate"]
            )
        
        # Calculate statistics
        if profile["vram_gb_samples"]:
            profile["avg_vram_gb"] = statistics.mean(profile["vram_gb_samples"])
            profile["max_vram_gb"] = max(profile["vram_gb_samples"])
            profile["vram_std_dev"] = statistics.stdev(profile["vram_gb_samples"]) if len(profile["vram_gb_samples"]) > 1 else 0
        
        if profile["generation_time_samples"]:
            profile["avg_generation_time"] = statistics.mean(profile["generation_time_samples"])
            profile["time_std_dev"] = statistics.stdev(profile["generation_time_samples"]) if len(profile["generation_time_samples"]) > 1 else 0
        
        if profile["gpu_utilization_samples"]:
            profile["avg_gpu_utilization"] = statistics.mean(profile["gpu_utilization_samples"])
        
        # Save profiles
        self._save_profiles()
    
    def get_model_estimate(
        self,
        model_name: str,
    ) -> Dict[str, float]:
        """Get resource estimates for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Resource estimates
        """
        if model_name not in self.model_profiles:
            # Return defaults if no profile
            return {
                "vram_gb": 8.0,  # Conservative default
                "generation_time": 60.0,
                "gpu_utilization": 50.0,
                "success_rate": 0.8,
            }
        
        profile = self.model_profiles[model_name]
        
        return {
            "vram_gb": profile.get("avg_vram_gb", 8.0),
            "vram_peak_gb": profile.get("max_vram_gb", 10.0),
            "generation_time": profile.get("avg_generation_time", 60.0),
            "gpu_utilization": profile.get("avg_gpu_utilization", 50.0),
            "success_rate": profile.get("success_rate", 0.8),
        }
    
    def _analyze_queue(
        self,
        queue_jobs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze queue to predict resource needs.
        
        Args:
            queue_jobs: List of pending jobs
            
        Returns:
            Queue analysis results
        """
        if not queue_jobs:
            return {
                "total_jobs": 0,
                "concurrent_models": 0,
                "model_counts": {},
                "estimated_vram_gb": 0,
                "estimated_time_minutes": 0,
                "bottleneck_models": [],
            }
        
        # Count models
        model_counts = defaultdict(int)
        for job in queue_jobs:
            model_name = job.get("model_name", "unknown")
            model_counts[model_name] += 1
        
        # Estimate concurrent models (based on VRAM constraints)
        total_vram = 24.0  # RTX 3090
        concurrent_models = 0
        used_vram = 0
        bottlenecks = []
        
        for model, count in model_counts.items():
            estimate = self.get_model_estimate(model)
            model_vram = estimate["vram_gb"]
            
            # How many can run concurrently?
            while count > 0 and used_vram + model_vram <= total_vram * 0.9:  # Leave 10% buffer
                concurrent_models += 1
                used_vram += model_vram
                count -= 1
            
            if count > 0:
                # This model is a bottleneck
                bottlenecks.append(model)
        
        # Estimate total time
        total_time = 0
        for model, count in model_counts.items():
            estimate = self.get_model_estimate(model)
            # Account for serialization if needed
            model_concurrent = min(count, max(1, int((total_vram * 0.9) / estimate["vram_gb"])))
            batches = (count + model_concurrent - 1) // model_concurrent
            total_time += batches * estimate["generation_time"] / 60  # Convert to minutes
        
        return {
            "total_jobs": len(queue_jobs),
            "concurrent_models": concurrent_models,
            "model_counts": dict(model_counts),
            "estimated_vram_gb": used_vram,
            "estimated_time_minutes": total_time,
            "bottleneck_models": bottlenecks,
        }
    
    def _forecast_vram(
        self,
        queue_analysis: Dict[str, Any],
        current_vram: float,
        horizon_minutes: int,
    ) -> Dict[str, Any]:
        """Forecast VRAM usage.
        
        Args:
            queue_analysis: Queue analysis results
            current_vram: Current VRAM usage
            horizon_minutes: Forecast horizon
            
        Returns:
            VRAM forecast
        """
        # Base prediction on queue analysis
        predicted_vram = queue_analysis["estimated_vram_gb"]
        
        # Add current baseline
        if current_vram > predicted_vram:
            # Some models already loaded
            predicted_vram = current_vram
        
        # Calculate confidence interval based on historical variance
        vram_variance = 2.0  # GB, estimated variance
        confidence_interval = (
            max(0, predicted_vram - vram_variance),
            min(24, predicted_vram + vram_variance),  # Cap at total VRAM
        )
        
        # Calculate probability of hitting peak (> 22GB)
        peak_threshold = 22.0
        if predicted_vram > peak_threshold:
            peak_probability = 1.0
        elif predicted_vram > peak_threshold - vram_variance:
            # Linear interpolation
            peak_probability = (predicted_vram - (peak_threshold - vram_variance)) / (2 * vram_variance)
        else:
            peak_probability = 0.0
        
        return {
            "predicted": predicted_vram,
            "confidence_interval": confidence_interval,
            "peak_probability": peak_probability,
        }
    
    def _forecast_gpu_utilization(
        self,
        queue_analysis: Dict[str, Any],
        current_utilization: float,
        horizon_minutes: int,
    ) -> Dict[str, Any]:
        """Forecast GPU utilization.
        
        Args:
            queue_analysis: Queue analysis results
            current_utilization: Current GPU utilization
            horizon_minutes: Forecast horizon
            
        Returns:
            GPU utilization forecast
        """
        # Estimate based on concurrent models
        concurrent = queue_analysis["concurrent_models"]
        
        # Each model typically uses 60-80% GPU when active
        avg_util_per_model = 70.0
        predicted_utilization = min(100, concurrent * avg_util_per_model / max(1, concurrent))
        
        # Blend with current if higher
        if current_utilization > predicted_utilization:
            # Decay current utilization over time
            decay_rate = 0.5  # 50% decay per hour
            decay_factor = decay_rate ** (horizon_minutes / 60)
            predicted_utilization = current_utilization * decay_factor + predicted_utilization * (1 - decay_factor)
        
        # Calculate overload probability (> 95%)
        overload_threshold = 95.0
        if predicted_utilization > overload_threshold:
            overload_probability = 1.0
        elif predicted_utilization > 80:
            # Linear interpolation
            overload_probability = (predicted_utilization - 80) / 15
        else:
            overload_probability = 0.0
        
        return {
            "predicted": predicted_utilization,
            "overload_probability": overload_probability,
        }
    
    def _estimate_completion_time(
        self,
        queue_jobs: List[Dict[str, Any]],
    ) -> float:
        """Estimate queue completion time.
        
        Args:
            queue_jobs: List of pending jobs
            
        Returns:
            Estimated completion time in minutes
        """
        if not queue_jobs:
            return 0.0
        
        # Group by model for better estimation
        model_jobs = defaultdict(list)
        for job in queue_jobs:
            model_name = job.get("model_name", "unknown")
            model_jobs[model_name].append(job)
        
        # Calculate parallel execution time
        max_time = 0.0
        for model, jobs in model_jobs.items():
            estimate = self.get_model_estimate(model)
            
            # Account for failures and retries
            effective_time = estimate["generation_time"] / estimate["success_rate"]
            
            # Total time for this model's jobs
            model_time = len(jobs) * effective_time / 60  # Convert to minutes
            
            max_time = max(max_time, model_time)
        
        return max_time
    
    def _assess_capacity(
        self,
        predicted_vram: float,
        predicted_gpu: float,
        concurrent_models: int,
    ) -> Dict[str, Any]:
        """Assess available capacity.
        
        Args:
            predicted_vram: Predicted VRAM usage
            predicted_gpu: Predicted GPU utilization
            concurrent_models: Number of concurrent models
            
        Returns:
            Capacity assessment
        """
        # Calculate available capacity
        vram_capacity = max(0, (24 - predicted_vram) / 24) * 100
        gpu_capacity = max(0, (100 - predicted_gpu) / 100) * 100
        
        # Overall capacity (minimum of both)
        available_percentage = min(vram_capacity, gpu_capacity)
        
        # Recommend batch size based on capacity
        if available_percentage > 50:
            recommended_batch_size = 4
        elif available_percentage > 30:
            recommended_batch_size = 2
        elif available_percentage > 10:
            recommended_batch_size = 1
        else:
            recommended_batch_size = 0
        
        # Can accept new jobs?
        can_accept = available_percentage > 10 and concurrent_models < 3
        
        return {
            "available_percentage": available_percentage,
            "vram_capacity": vram_capacity,
            "gpu_capacity": gpu_capacity,
            "recommended_batch_size": recommended_batch_size,
            "can_accept_jobs": can_accept,
        }
    
    def _assess_risk(
        self,
        vram_forecast: Dict[str, Any],
        gpu_forecast: Dict[str, Any],
        capacity: Dict[str, Any],
        queue_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess risk levels and provide recommendations.
        
        Args:
            vram_forecast: VRAM forecast results
            gpu_forecast: GPU forecast results
            capacity: Capacity assessment
            queue_analysis: Queue analysis results
            
        Returns:
            Risk assessment
        """
        risk_factors = []
        recommendations = []
        risk_score = 0
        
        # Check VRAM risk
        if vram_forecast["peak_probability"] > 0.7:
            risk_factors.append("High probability of VRAM exhaustion")
            recommendations.append("Consider reducing batch sizes or using memory optimization")
            risk_score += 3
        elif vram_forecast["peak_probability"] > 0.3:
            risk_factors.append("Moderate VRAM pressure expected")
            recommendations.append("Monitor VRAM usage closely")
            risk_score += 1
        
        # Check GPU overload risk
        if gpu_forecast["overload_probability"] > 0.7:
            risk_factors.append("GPU likely to be overloaded")
            recommendations.append("Reduce concurrent model executions")
            risk_score += 2
        elif gpu_forecast["overload_probability"] > 0.3:
            risk_factors.append("Moderate GPU load expected")
            risk_score += 1
        
        # Check capacity
        if capacity["available_percentage"] < 10:
            risk_factors.append("Very low available capacity")
            recommendations.append("Delay non-critical jobs")
            risk_score += 3
        elif capacity["available_percentage"] < 30:
            risk_factors.append("Limited capacity available")
            recommendations.append("Prioritize important jobs")
            risk_score += 1
        
        # Check bottlenecks
        if queue_analysis["bottleneck_models"]:
            risk_factors.append(f"Model bottlenecks: {', '.join(queue_analysis['bottleneck_models'])}")
            recommendations.append("Consider load balancing or model optimization")
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Add general recommendations
        if risk_level == "high":
            recommendations.insert(0, "System under high load - consider pausing new jobs")
        elif risk_level == "low" and capacity["available_percentage"] > 50:
            recommendations.append("System has spare capacity - safe to add more jobs")
        
        return {
            "level": risk_level,
            "score": risk_score,
            "factors": risk_factors,
            "recommendations": recommendations,
        }
    
    def _save_profiles(self) -> None:
        """Save model profiles to disk."""
        try:
            file_path = self.storage_dir / "model_profiles.json"
            
            # Convert to serializable format
            profiles_data = {}
            for model, profile in self.model_profiles.items():
                profiles_data[model] = {
                    k: v for k, v in profile.items()
                    if not k.endswith("_samples")  # Don't save raw samples
                }
            
            with file_path.open("w") as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
    
    def _load_history(self) -> None:
        """Load historical data."""
        try:
            # Load model profiles
            profiles_file = self.storage_dir / "model_profiles.json"
            if profiles_file.exists():
                with profiles_file.open() as f:
                    profiles_data = json.load(f)
                    
                for model, data in profiles_data.items():
                    self.model_profiles[model] = data
                    # Initialize sample lists
                    self.model_profiles[model]["vram_gb_samples"] = []
                    self.model_profiles[model]["generation_time_samples"] = []
                    self.model_profiles[model]["gpu_utilization_samples"] = []
                
                logger.info(f"Loaded profiles for {len(self.model_profiles)} models")
                
        except Exception as e:
            logger.error(f"Failed to load history: {e}")