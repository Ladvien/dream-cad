"""Cost analysis for multi-model 3D generation system."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CostReport:
    """Cost analysis report."""
    
    report_date: datetime
    period_start: datetime
    period_end: datetime
    
    # Total costs
    total_cost: float
    compute_cost: float
    storage_cost: float
    
    # Per-model costs
    model_costs: Dict[str, float]
    model_cost_breakdown: Dict[str, Dict[str, float]]  # Model -> cost components
    
    # Cost per output
    avg_cost_per_generation: float
    cost_per_successful_generation: float
    cost_per_quality_point: float
    
    # Resource utilization costs
    gpu_hours: float
    gpu_cost: float
    vram_gb_hours: float
    vram_cost: float
    
    # Efficiency metrics
    cost_efficiency_score: float
    most_cost_effective_model: str
    least_cost_effective_model: str
    
    # Cost trends
    daily_costs: List[float]
    hourly_cost_distribution: Dict[int, float]  # Hour -> avg cost
    
    # Cost optimization
    optimization_potential: float  # Potential savings
    optimization_recommendations: List[str]
    
    # Projections
    projected_monthly_cost: float
    projected_annual_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["report_date"] = self.report_date.isoformat()
        data["period_start"] = self.period_start.isoformat()
        data["period_end"] = self.period_end.isoformat()
        return data


class CostAnalyzer:
    """Analyze costs for multi-model 3D generation."""
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        cost_config: Optional[Dict[str, float]] = None,
    ):
        """Initialize cost analyzer.
        
        Args:
            storage_dir: Directory to store reports
            cost_config: Cost configuration (per-unit costs)
        """
        self.storage_dir = storage_dir or Path("monitoring/costs")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Default cost configuration (can be customized)
        self.cost_config = cost_config or {
            "gpu_hour": 0.50,  # $ per GPU hour
            "vram_gb_hour": 0.02,  # $ per GB-hour of VRAM
            "storage_gb_month": 0.10,  # $ per GB per month
            "electricity_kwh": 0.12,  # $ per kWh
            "gpu_power_watts": 350,  # RTX 3090 typical power
        }
        
        # Model-specific cost adjustments
        self.model_cost_multipliers: Dict[str, float] = {}
    
    def generate_report(
        self,
        metrics_data: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime,
    ) -> CostReport:
        """Generate cost analysis report.
        
        Args:
            metrics_data: List of generation metrics
            period_start: Start of analysis period
            period_end: End of analysis period
            
        Returns:
            Cost analysis report
        """
        # Calculate base costs
        compute_costs = self._calculate_compute_costs(metrics_data)
        storage_costs = self._calculate_storage_costs(metrics_data, period_start, period_end)
        
        # Per-model analysis
        model_analysis = self._analyze_model_costs(metrics_data)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            metrics_data,
            compute_costs["total"],
        )
        
        # Analyze trends
        trends = self._analyze_cost_trends(metrics_data, period_start, period_end)
        
        # Find optimization opportunities
        optimizations = self._find_optimization_opportunities(
            model_analysis,
            efficiency_metrics,
        )
        
        # Generate projections
        projections = self._generate_projections(
            compute_costs["total"] + storage_costs,
            period_start,
            period_end,
        )
        
        # Determine most/least cost-effective models
        if model_analysis["model_costs"]:
            sorted_models = sorted(
                model_analysis["cost_per_quality"].items(),
                key=lambda x: x[1],
            )
            most_effective = sorted_models[0][0] if sorted_models else "N/A"
            least_effective = sorted_models[-1][0] if sorted_models else "N/A"
        else:
            most_effective = least_effective = "N/A"
        
        # Create report
        report = CostReport(
            report_date=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            total_cost=compute_costs["total"] + storage_costs,
            compute_cost=compute_costs["total"],
            storage_cost=storage_costs,
            model_costs=model_analysis["model_costs"],
            model_cost_breakdown=model_analysis["breakdown"],
            avg_cost_per_generation=efficiency_metrics["avg_cost_per_generation"],
            cost_per_successful_generation=efficiency_metrics["cost_per_success"],
            cost_per_quality_point=efficiency_metrics["cost_per_quality"],
            gpu_hours=compute_costs["gpu_hours"],
            gpu_cost=compute_costs["gpu_cost"],
            vram_gb_hours=compute_costs["vram_gb_hours"],
            vram_cost=compute_costs["vram_cost"],
            cost_efficiency_score=efficiency_metrics["efficiency_score"],
            most_cost_effective_model=most_effective,
            least_cost_effective_model=least_effective,
            daily_costs=trends["daily_costs"],
            hourly_cost_distribution=trends["hourly_distribution"],
            optimization_potential=optimizations["potential_savings"],
            optimization_recommendations=optimizations["recommendations"],
            projected_monthly_cost=projections["monthly"],
            projected_annual_cost=projections["annual"],
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _calculate_compute_costs(
        self,
        metrics_data: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate compute costs from metrics.
        
        Args:
            metrics_data: List of generation metrics
            
        Returns:
            Compute cost breakdown
        """
        total_gpu_hours = 0
        total_vram_gb_hours = 0
        electricity_kwh = 0
        
        for metric in metrics_data:
            # GPU time
            gen_time_hours = metric.get("generation_time_seconds", 0) / 3600
            total_gpu_hours += gen_time_hours
            
            # VRAM usage
            vram_gb = metric.get("vram_peak_gb", 0)
            total_vram_gb_hours += vram_gb * gen_time_hours
            
            # Electricity (simplified)
            power_kw = self.cost_config["gpu_power_watts"] / 1000
            electricity_kwh += power_kw * gen_time_hours
        
        # Calculate costs
        gpu_cost = total_gpu_hours * self.cost_config["gpu_hour"]
        vram_cost = total_vram_gb_hours * self.cost_config["vram_gb_hour"]
        electricity_cost = electricity_kwh * self.cost_config["electricity_kwh"]
        
        return {
            "gpu_hours": total_gpu_hours,
            "gpu_cost": gpu_cost,
            "vram_gb_hours": total_vram_gb_hours,
            "vram_cost": vram_cost,
            "electricity_cost": electricity_cost,
            "total": gpu_cost + vram_cost + electricity_cost,
        }
    
    def _calculate_storage_costs(
        self,
        metrics_data: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime,
    ) -> float:
        """Calculate storage costs.
        
        Args:
            metrics_data: List of generation metrics
            period_start: Start of period
            period_end: End of period
            
        Returns:
            Storage cost
        """
        # Estimate storage usage (simplified)
        # Assume each generation produces ~100MB of data
        num_generations = len(metrics_data)
        storage_gb = (num_generations * 100) / 1024  # MB to GB
        
        # Calculate monthly fraction
        period_days = (period_end - period_start).days
        monthly_fraction = period_days / 30
        
        # Storage cost
        storage_cost = storage_gb * self.cost_config["storage_gb_month"] * monthly_fraction
        
        return storage_cost
    
    def _analyze_model_costs(
        self,
        metrics_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze costs per model.
        
        Args:
            metrics_data: List of generation metrics
            
        Returns:
            Model cost analysis
        """
        model_costs = defaultdict(float)
        model_breakdown = defaultdict(lambda: defaultdict(float))
        model_quality = defaultdict(list)
        model_success = defaultdict(lambda: {"success": 0, "total": 0})
        
        for metric in metrics_data:
            model = metric.get("model_name", "unknown")
            
            # Calculate cost for this generation
            gen_time_hours = metric.get("generation_time_seconds", 0) / 3600
            vram_gb = metric.get("vram_peak_gb", 0)
            
            gpu_cost = gen_time_hours * self.cost_config["gpu_hour"]
            vram_cost = vram_gb * gen_time_hours * self.cost_config["vram_gb_hour"]
            
            # Apply model-specific multiplier if exists
            multiplier = self.model_cost_multipliers.get(model, 1.0)
            total_cost = (gpu_cost + vram_cost) * multiplier
            
            model_costs[model] += total_cost
            model_breakdown[model]["gpu"] += gpu_cost
            model_breakdown[model]["vram"] += vram_cost
            model_breakdown[model]["total"] += total_cost
            
            # Track quality and success
            if metric.get("quality_score", 0) > 0:
                model_quality[model].append(metric["quality_score"])
            
            model_success[model]["total"] += 1
            if metric.get("success", False):
                model_success[model]["success"] += 1
        
        # Calculate cost per quality point
        cost_per_quality = {}
        for model in model_costs:
            avg_quality = (
                sum(model_quality[model]) / len(model_quality[model])
                if model_quality[model] else 50
            )
            if avg_quality > 0:
                cost_per_quality[model] = model_costs[model] / avg_quality
            else:
                cost_per_quality[model] = float('inf')
        
        return {
            "model_costs": dict(model_costs),
            "breakdown": dict(model_breakdown),
            "cost_per_quality": cost_per_quality,
            "model_success": dict(model_success),
        }
    
    def _calculate_efficiency_metrics(
        self,
        metrics_data: List[Dict[str, Any]],
        total_cost: float,
    ) -> Dict[str, float]:
        """Calculate cost efficiency metrics.
        
        Args:
            metrics_data: List of generation metrics
            total_cost: Total cost for period
            
        Returns:
            Efficiency metrics
        """
        total_generations = len(metrics_data)
        successful_generations = sum(
            1 for m in metrics_data if m.get("success", False)
        )
        
        # Average costs
        avg_cost_per_generation = (
            total_cost / total_generations if total_generations > 0 else 0
        )
        cost_per_success = (
            total_cost / successful_generations if successful_generations > 0 else 0
        )
        
        # Quality-adjusted cost
        total_quality = sum(
            m.get("quality_score", 0) for m in metrics_data
            if m.get("success", False)
        )
        cost_per_quality = (
            total_cost / total_quality if total_quality > 0 else float('inf')
        )
        
        # Efficiency score (0-100)
        # Based on success rate and quality per dollar
        success_rate = successful_generations / total_generations if total_generations > 0 else 0
        quality_per_dollar = 1 / cost_per_quality if cost_per_quality > 0 else 0
        
        efficiency_score = min(100, (
            success_rate * 50 +  # 50% weight on success
            min(50, quality_per_dollar * 100)  # 50% weight on quality/cost
        ))
        
        return {
            "avg_cost_per_generation": avg_cost_per_generation,
            "cost_per_success": cost_per_success,
            "cost_per_quality": cost_per_quality,
            "efficiency_score": efficiency_score,
        }
    
    def _analyze_cost_trends(
        self,
        metrics_data: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime,
    ) -> Dict[str, Any]:
        """Analyze cost trends over time.
        
        Args:
            metrics_data: List of generation metrics
            period_start: Start of period
            period_end: End of period
            
        Returns:
            Cost trend analysis
        """
        # Group costs by day
        daily_costs = defaultdict(float)
        hourly_costs = defaultdict(list)
        
        for metric in metrics_data:
            timestamp = metric.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                continue
            
            # Calculate cost for this metric
            gen_time_hours = metric.get("generation_time_seconds", 0) / 3600
            vram_gb = metric.get("vram_peak_gb", 0)
            
            cost = (
                gen_time_hours * self.cost_config["gpu_hour"] +
                vram_gb * gen_time_hours * self.cost_config["vram_gb_hour"]
            )
            
            # Add to daily total
            date = timestamp.date()
            daily_costs[date] += cost
            
            # Add to hourly distribution
            hour = timestamp.hour
            hourly_costs[hour].append(cost)
        
        # Convert to lists
        period_days = (period_end - period_start).days
        daily_cost_list = []
        current_date = period_start.date()
        
        for _ in range(period_days):
            daily_cost_list.append(daily_costs.get(current_date, 0))
            current_date += timedelta(days=1)
        
        # Calculate hourly distribution
        hourly_distribution = {}
        for hour in range(24):
            costs = hourly_costs.get(hour, [])
            hourly_distribution[hour] = sum(costs) / len(costs) if costs else 0
        
        return {
            "daily_costs": daily_cost_list,
            "hourly_distribution": hourly_distribution,
        }
    
    def _find_optimization_opportunities(
        self,
        model_analysis: Dict[str, Any],
        efficiency_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Find cost optimization opportunities.
        
        Args:
            model_analysis: Model cost analysis
            efficiency_metrics: Efficiency metrics
            
        Returns:
            Optimization opportunities
        """
        recommendations = []
        potential_savings = 0
        
        # Check for expensive models
        if model_analysis["model_costs"]:
            avg_cost = sum(model_analysis["model_costs"].values()) / len(model_analysis["model_costs"])
            
            for model, cost in model_analysis["model_costs"].items():
                if cost > avg_cost * 1.5:
                    recommendations.append(
                        f"Consider optimizing or replacing {model} (cost {cost:.2f} vs avg {avg_cost:.2f})"
                    )
                    potential_savings += (cost - avg_cost) * 0.3  # 30% potential reduction
        
        # Check efficiency score
        if efficiency_metrics["efficiency_score"] < 60:
            recommendations.append(
                "Low overall efficiency - review model configurations and error rates"
            )
            potential_savings += efficiency_metrics["avg_cost_per_generation"] * 0.2
        
        # Check success rate impact
        if efficiency_metrics["cost_per_success"] > efficiency_metrics["avg_cost_per_generation"] * 1.2:
            recommendations.append(
                "High failure rate increasing costs - investigate and fix errors"
            )
            savings = efficiency_metrics["cost_per_success"] - efficiency_metrics["avg_cost_per_generation"]
            potential_savings += savings * 0.5
        
        # Time-based optimization
        recommendations.append(
            "Consider scheduling non-urgent jobs during off-peak hours for potential discounts"
        )
        
        # Batch processing
        recommendations.append(
            "Batch similar jobs together to reduce model loading/unloading overhead"
        )
        
        return {
            "recommendations": recommendations[:5],  # Top 5 recommendations
            "potential_savings": potential_savings,
        }
    
    def _generate_projections(
        self,
        period_cost: float,
        period_start: datetime,
        period_end: datetime,
    ) -> Dict[str, float]:
        """Generate cost projections.
        
        Args:
            period_cost: Cost for the analysis period
            period_start: Start of period
            period_end: End of period
            
        Returns:
            Cost projections
        """
        # Calculate daily rate
        period_days = max(1, (period_end - period_start).days)
        daily_rate = period_cost / period_days
        
        # Project monthly and annual
        monthly_projection = daily_rate * 30
        annual_projection = daily_rate * 365
        
        # Apply growth factor (assume 10% growth)
        growth_factor = 1.1
        annual_projection *= growth_factor
        
        return {
            "monthly": monthly_projection,
            "annual": annual_projection,
        }
    
    def _save_report(self, report: CostReport) -> None:
        """Save cost report to disk.
        
        Args:
            report: Cost report to save
        """
        try:
            filename = f"cost_report_{report.report_date.strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.storage_dir / filename
            
            with file_path.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.info(f"Saved cost report to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cost report: {e}")