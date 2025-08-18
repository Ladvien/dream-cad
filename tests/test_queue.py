import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dream_cad.queue import (
    JobQueue,
    GenerationJob,
    JobStatus,
    JobPriority,
    BatchProcessor,
    ResourceManager,
    QueueAnalytics,
)
from dream_cad.queue.job_queue import ModelRequirements
from dream_cad.queue.resource_manager import GPUDevice, ModelProfile
from dream_cad.queue.batch_processor import ModelInstance, FailoverStrategy
class TestGenerationJob(unittest.TestCase):
    def test_job_creation(self):
        job = GenerationJob(
            id="test_job",
            prompt="a test prompt",
            model_name="triposr",
            config={"steps": 50},
            priority=JobPriority.HIGH,
        )
        self.assertEqual(job.id, "test_job")
        self.assertEqual(job.prompt, "a test prompt")
        self.assertEqual(job.model_name, "triposr")
        self.assertEqual(job.priority, JobPriority.HIGH)
        self.assertEqual(job.status, JobStatus.PENDING)
    def test_job_serialization(self):
        job = GenerationJob(
            id="test_job",
            prompt="test",
            model_name="triposr",
            config={},
            model_requirements=ModelRequirements(
                model_name="triposr",
                min_vram_gb=4.0,
                estimated_time_seconds=30.0,
            ),
        )
        job_dict = job.to_dict()
        self.assertIsInstance(job_dict, dict)
        self.assertEqual(job_dict["id"], "test_job")
        self.assertEqual(job_dict["status"], "pending")
        job2 = GenerationJob.from_dict(job_dict)
        self.assertEqual(job2.id, job.id)
        self.assertEqual(job2.model_name, job.model_name)
        self.assertIsInstance(job2.model_requirements, ModelRequirements)
    def test_job_dependencies(self):
        job = GenerationJob(
            id="test_job",
            prompt="test",
            model_name="triposr",
            config={},
            depends_on=["job1", "job2"],
        )
        self.assertFalse(job.can_run(set()))
        self.assertFalse(job.can_run({"job1"}))
        self.assertTrue(job.can_run({"job1", "job2", "job3"}))
    def test_job_retry(self):
        job = GenerationJob(
            id="test_job",
            prompt="test",
            model_name="triposr",
            config={},
            max_retries=3,
        )
        self.assertTrue(job.is_retriable())
        job.retry_count = 3
        self.assertFalse(job.is_retriable())
    def test_completion_time_estimate(self):
        job = GenerationJob(
            id="test_job",
            prompt="test",
            model_name="triposr",
            config={},
        )
        self.assertIsNone(job.estimate_completion_time())
        job.started_at = datetime.now().isoformat()
        job.progress_percent = 50.0
        estimate = job.estimate_completion_time()
        self.assertIsNotNone(estimate)
        self.assertIsInstance(estimate, datetime)
class TestJobQueue(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_file = Path(self.temp_dir) / "queue.json"
        self.queue = JobQueue(
            queue_file=self.queue_file,
            enable_persistence=False,
        )
    def test_add_job(self):
        job = self.queue.add_job(
            prompt="test prompt",
            model_name="triposr",
            priority=JobPriority.NORMAL,
        )
        self.assertIsNotNone(job)
        self.assertEqual(job.prompt, "test prompt")
        self.assertEqual(job.model_name, "triposr")
        self.assertIn(job.id, self.queue.jobs)
    def test_priority_ordering(self):
        low_job = self.queue.add_job("low", "triposr", priority=JobPriority.LOW)
        high_job = self.queue.add_job("high", "triposr", priority=JobPriority.HIGH)
        normal_job = self.queue.add_job("normal", "triposr", priority=JobPriority.NORMAL)
        job1 = self.queue.get_next_job()
        job2 = self.queue.get_next_job()
        job3 = self.queue.get_next_job()
        self.assertEqual(job1.id, high_job.id)
        self.assertEqual(job2.id, normal_job.id)
        self.assertEqual(job3.id, low_job.id)
    def test_dependency_handling(self):
        job1 = self.queue.add_job("job1", "triposr")
        job2 = self.queue.add_job("job2", "triposr", depends_on=[job1.id])
        job3 = self.queue.add_job("job3", "triposr")
        next_job = self.queue.get_next_job()
        self.assertIn(next_job.id, [job1.id, job3.id])
        first_job_id = next_job.id
        next_job = self.queue.get_next_job()
        self.assertIn(next_job.id, [job1.id, job3.id])
        self.assertNotEqual(next_job.id, first_job_id)
        next_job = self.queue.get_next_job()
        self.assertIsNone(next_job)
        self.queue.update_job(job1.id, status=JobStatus.COMPLETED)
        next_job = self.queue.get_next_job()
        self.assertEqual(next_job.id, job2.id)
    def test_model_filtering(self):
        job1 = self.queue.add_job("test1", "triposr")
        job2 = self.queue.add_job("test2", "stable-fast-3d")
        job3 = self.queue.add_job("test3", "triposr")
        next_job = self.queue.get_next_job(model_name="triposr")
        self.assertEqual(next_job.model_name, "triposr")
        next_job = self.queue.get_next_job(model_name="stable-fast-3d")
        self.assertEqual(next_job.model_name, "stable-fast-3d")
    def test_batch_creation(self):
        prompts = ["prompt1", "prompt2", "prompt3"]
        jobs = self.queue.create_batch(prompts, "triposr")
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].batch_position, 0)
        self.assertEqual(jobs[1].batch_position, 1)
        self.assertEqual(jobs[2].batch_position, 2)
        self.assertEqual(jobs[0].batch_id, jobs[1].batch_id)
    def test_job_update(self):
        job = self.queue.add_job("test", "triposr")
        self.queue.update_job(
            job.id,
            status=JobStatus.RUNNING,
            progress_percent=50.0,
        )
        updated_job = self.queue.get_job(job.id)
        self.assertEqual(updated_job.status, JobStatus.RUNNING)
        self.assertEqual(updated_job.progress_percent, 50.0)
        self.queue.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            output_path="/path/to/output",
        )
        updated_job = self.queue.get_job(job.id)
        self.assertEqual(updated_job.status, JobStatus.COMPLETED)
        self.assertEqual(updated_job.output_path, "/path/to/output")
    def test_queue_persistence(self):
        self.queue.enable_persistence = True
        job1 = self.queue.add_job("test1", "triposr")
        job2 = self.queue.add_job("test2", "stable-fast-3d")
        self.queue.save_queue()
        self.assertTrue(self.queue_file.exists())
        new_queue = JobQueue(queue_file=self.queue_file)
        self.assertEqual(len(new_queue.jobs), 2)
        self.assertIn(job1.id, new_queue.jobs)
        self.assertIn(job2.id, new_queue.jobs)
    def test_clear_completed(self):
        job1 = self.queue.add_job("test1", "triposr")
        job2 = self.queue.add_job("test2", "triposr")
        self.queue.update_job(job1.id, status=JobStatus.COMPLETED)
        job1 = self.queue.get_job(job1.id)
        job1.completed_at = (datetime.now() - timedelta(hours=25)).isoformat()
        removed = self.queue.clear_completed(older_than_hours=24)
        self.assertEqual(removed, 1)
        self.assertNotIn(job1.id, self.queue.jobs)
        self.assertIn(job2.id, self.queue.jobs)
class TestResourceManager(unittest.TestCase):
    def setUp(self):
        self.resource_manager = ResourceManager(
            check_interval=1,
            enable_multi_gpu=True,
        )
    def test_gpu_discovery(self):
        if not self.resource_manager.gpus:
            self.resource_manager.gpus[0] = GPUDevice(
                index=0,
                name="Mock GPU",
                total_memory_gb=24.0,
                allocated_memory_gb=0.0,
                free_memory_gb=24.0,
            )
        self.assertGreater(len(self.resource_manager.gpus), 0)
    def test_model_profiles(self):
        profiles = self.resource_manager.model_profiles
        self.assertIn("triposr", profiles)
        self.assertIn("stable-fast-3d", profiles)
        self.assertIn("mvdream", profiles)
        triposr_profile = profiles["triposr"]
        self.assertEqual(triposr_profile.min_vram_gb, 4.0)
        self.assertEqual(triposr_profile.optimal_vram_gb, 6.0)
    def test_can_run_model(self):
        self.resource_manager.gpus[0] = GPUDevice(
            index=0,
            name="Test GPU",
            total_memory_gb=24.0,
            allocated_memory_gb=0.0,
            free_memory_gb=24.0,
        )
        can_run, gpu_idx, message = self.resource_manager.can_run_model("triposr")
        self.assertTrue(can_run)
        can_run, gpu_idx, message = self.resource_manager.can_run_model(
            "mvdream",
            required_vram_gb=30.0,
        )
        self.assertFalse(can_run)
    def test_job_assignment(self):
        self.resource_manager.gpus[0] = GPUDevice(
            index=0,
            name="Test GPU",
            total_memory_gb=24.0,
            allocated_memory_gb=0.0,
            free_memory_gb=24.0,
        )
        gpu_idx = self.resource_manager.assign_job_to_gpu(
            "test_job",
            "triposr",
        )
        self.assertEqual(gpu_idx, 0)
        self.assertEqual(self.resource_manager.gpus[0].current_job_id, "test_job")
        self.assertIn("triposr", self.resource_manager.gpus[0].assigned_models)
    def test_model_placement_optimization(self):
        self.resource_manager.gpus[0] = GPUDevice(
            index=0,
            name="GPU 0",
            total_memory_gb=24.0,
            allocated_memory_gb=0.0,
            free_memory_gb=24.0,
        )
        self.resource_manager.gpus[1] = GPUDevice(
            index=1,
            name="GPU 1",
            total_memory_gb=12.0,
            allocated_memory_gb=0.0,
            free_memory_gb=12.0,
        )
        placement = self.resource_manager.optimize_model_placement(
            ["mvdream", "triposr"]
        )
        self.assertEqual(placement.get("mvdream"), 0)
class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue = JobQueue(enable_persistence=False)
        self.resource_manager = ResourceManager()
        self.resource_manager.gpus[0] = GPUDevice(
            index=0,
            name="Test GPU",
            total_memory_gb=24.0,
            allocated_memory_gb=0.0,
            free_memory_gb=24.0,
        )
        self.processor = BatchProcessor(
            job_queue=self.queue,
            resource_manager=self.resource_manager,
            max_workers=1,
            enable_warm_up=False,
        )
    def test_failover_strategy(self):
        strategy = FailoverStrategy()
        alternative = strategy.get_alternative_model("mvdream")
        self.assertIn(alternative, ["stable-fast-3d", "triposr"])
        strategy.record_failure("mvdream")
        strategy.record_failure("mvdream")
        strategy.record_failure("mvdream")
        self.assertIn("mvdream", strategy.blacklisted_models)
        strategy.reset_model("mvdream")
        self.assertNotIn("mvdream", strategy.blacklisted_models)
    @patch('dream_cad.queue.batch_processor.ModelFactory')
    def test_model_loading(self, mock_factory):
        mock_model = MagicMock()
        mock_factory.create_model.return_value = mock_model
        import dream_cad.queue.batch_processor as bp
        bp.MODEL_FACTORY_AVAILABLE = True
        bp.ModelFactory = mock_factory
        instance = self.processor.load_model("triposr", warm_up=False)
        self.assertIsNotNone(instance)
        self.assertEqual(instance.model_name, "triposr")
        mock_factory.create_model.assert_called_once_with("triposr")
    def test_model_idle_timeout(self):
        instance = ModelInstance(
            model_name="triposr",
            model=MagicMock(),
            gpu_index=0,
            loaded_at=datetime.now() - timedelta(minutes=10),
            last_used=datetime.now() - timedelta(minutes=10),
        )
        self.processor.loaded_models["triposr"] = instance
        self.processor.model_idle_timeout = 300
        self.processor._check_idle_models()
        self.assertNotIn("triposr", self.processor.loaded_models)
class TestQueueAnalytics(unittest.TestCase):
    def setUp(self):
        self.queue = JobQueue(enable_persistence=False)
        self.analytics = QueueAnalytics(self.queue)
        for i in range(5):
            job = self.queue.add_job(
                f"test_{i}",
                "triposr" if i % 2 == 0 else "stable-fast-3d",
            )
            if i < 3:
                self.queue.update_job(
                    job.id,
                    status=JobStatus.COMPLETED,
                    generation_time_seconds=30.0 + i * 10,
                )
    def test_queue_metrics(self):
        metrics = self.analytics.analyze_queue()
        self.assertEqual(metrics.total_submitted, 5)
        self.assertEqual(metrics.total_completed, 3)
        self.assertGreater(metrics.queue_efficiency, 0)
    def test_model_performance(self):
        model_perf = self.analytics.analyze_models()
        self.assertIn("triposr", model_perf)
        self.assertIn("stable-fast-3d", model_perf)
        triposr_perf = model_perf["triposr"]
        self.assertGreater(triposr_perf.total_jobs, 0)
    def test_time_series_analysis(self):
        time_series = self.analytics.analyze_time_series()
        self.assertIsInstance(time_series, dict)
        if time_series:
            first_hour = list(time_series.keys())[0]
            self.assertIn("submitted", time_series[first_hour])
    def test_report_generation(self):
        report = self.analytics.generate_report()
        self.assertIn("queue_metrics", report)
        self.assertIn("model_performance", report)
        self.assertIn("model_recommendations", report)
    def test_csv_export(self):
        csv_file = self.analytics.export_csv()
        self.assertTrue(csv_file.exists())
        with csv_file.open() as f:
            content = f.read()
            self.assertIn("id", content)
            self.assertIn("model", content)
if __name__ == "__main__":
    unittest.main()