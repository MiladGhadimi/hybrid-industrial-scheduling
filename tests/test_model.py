import json
import tempfile
import unittest
from pathlib import Path

from hybrid_scheduler.generator import generate_instance
from hybrid_scheduler.model import Job, SchedulingInstance


class ModelTests(unittest.TestCase):
    def test_generated_instance_is_reproducible(self):
        self.assertEqual(generate_instance(5, 7).to_dict(), generate_instance(5, 7).to_dict())

    def test_evaluate_rejects_non_permutation(self):
        instance = generate_instance(4, 1)
        with self.assertRaises(ValueError):
            instance.evaluate([0, 1, 1, 3])

    def test_cost_decomposition(self):
        instance = generate_instance(4, 2)
        parts = instance.evaluate(range(4))
        self.assertAlmostEqual(parts["objective"], parts["placement_cost"] + parts["changeover_cost"])

    def test_validation_rejects_missing_family(self):
        jobs = (Job("J1", "A", 0, 1, 1), Job("J2", "B", 1, 1, 1))
        with self.assertRaises(ValueError):
            SchedulingInstance(jobs, (1, 1), {"A": {"A": 0, "B": 1}})

    def test_legacy_instances_with_processing_time_still_load(self):
        """v0.1.0 wrote a processing_time field that never entered any cost term."""
        payload = generate_instance(4, 3).to_dict()
        for job in payload["jobs"]:
            job["processing_time"] = 2.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = SchedulingInstance.load(path)
        self.assertEqual(loaded.size, 4)
        self.assertFalse(hasattr(loaded.jobs[0], "processing_time"))

    def test_unknown_job_field_is_rejected(self):
        payload = generate_instance(4, 3).to_dict()
        payload["jobs"][0]["prioroty"] = 2.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prioroty"):
                SchedulingInstance.load(path)

    def test_round_trip_save_and_load(self):
        instance = generate_instance(6, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instance.json"
            instance.save(path)
            reloaded = SchedulingInstance.load(path)
        self.assertEqual(reloaded.to_dict(), instance.to_dict())
        self.assertAlmostEqual(
            reloaded.evaluate(range(6))["objective"], instance.evaluate(range(6))["objective"]
        )


if __name__ == "__main__":
    unittest.main()
