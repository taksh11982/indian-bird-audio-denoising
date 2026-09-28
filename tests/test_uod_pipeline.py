import unittest

from uod_pipeline import (
    SampleRecord,
    average_precision_binary,
    compare_before_after_map,
    detect_species_outliers,
    macro_map,
    prune_records,
    species_error_variance,
)


class UODPipelineTests(unittest.TestCase):
    def test_species_outlier_detection_is_species_aware(self):
        labels = ["sp1", "sp1", "sp1", "sp1", "sp1", "sp2", "sp2", "sp2", "sp2", "sp2"]
        errors = [0.2, 0.25, 0.21, 0.22, 2.8, 0.4, 0.42, 0.45, 0.39, 0.41]

        outliers = detect_species_outliers(errors, labels, z_threshold=3.0, min_group_size=5)

        self.assertEqual(outliers, [False, False, False, False, True, False, False, False, False, False])

    def test_prune_records_removes_flagged_items(self):
        records = [
            SampleRecord("a", "sp1"),
            SampleRecord("b", "sp1"),
            SampleRecord("c", "sp2"),
        ]
        pruned = prune_records(records, [False, True, False])

        self.assertEqual([record.sample_id for record in pruned], ["a", "c"])

    def test_binary_average_precision_perfect_ranking(self):
        ap = average_precision_binary([1, 0, 1, 0], [0.9, 0.4, 0.8, 0.1])
        self.assertAlmostEqual(ap, 1.0, places=8)

    def test_macro_map_and_before_after_gain(self):
        labels = ["a", "a", "b", "b"]
        class_order = ["a", "b"]

        baseline_probs = [
            {"a": 0.51, "b": 0.49},
            {"a": 0.50, "b": 0.50},
            {"a": 0.53, "b": 0.47},
            {"a": 0.52, "b": 0.48},
        ]
        denoised_probs = [
            {"a": 0.95, "b": 0.05},
            {"a": 0.9, "b": 0.1},
            {"a": 0.1, "b": 0.9},
            {"a": 0.05, "b": 0.95},
        ]

        baseline_map = macro_map(labels, baseline_probs, class_order)
        summary = compare_before_after_map(labels, baseline_probs, denoised_probs, class_order)

        self.assertGreater(summary["denoised_map"], baseline_map)
        self.assertGreater(summary["delta_map"], 0.0)

    def test_species_variance_summary(self):
        labels = ["a", "a", "b", "b"]
        errors = [0.1, 0.3, 0.2, 0.2]

        variances = species_error_variance(errors, labels)

        self.assertAlmostEqual(variances["a"], 0.01)
        self.assertAlmostEqual(variances["b"], 0.0)


if __name__ == "__main__":
    unittest.main()
