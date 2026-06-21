import unittest

from scripts.evaluate_gnc_mc import parse_seeds, reduce_metrics


class TestGncMonteCarloMetrics(unittest.TestCase):
    def test_parse_seeds_supports_ranges_and_lists(self):
        self.assertEqual(parse_seeds("0:3,10,12:16:2"), [0, 1, 2, 10, 12, 14])

    def test_reduce_metrics_computes_bottom_ten_percent_and_success(self):
        rows = [
            {
                "variant": "a",
                "seed": seed,
                "popped_count": count,
                "target_switch_count": seed,
                "mean_target_dwell_time": 0.5,
                "error": "",
            }
            for seed, count in enumerate(range(1, 11))
        ]

        summary = reduce_metrics(rows, [5, 10])
        metrics = summary["a"]

        self.assertEqual(metrics["valid_cases"], 10)
        self.assertEqual(metrics["min"], 1)
        self.assertEqual(metrics["max"], 10)
        self.assertEqual(metrics["bottom_10_percent_mean"], 1.0)
        self.assertEqual(metrics["success_rate@5"], 0.6)
        self.assertEqual(metrics["success_rate@10"], 0.1)


if __name__ == "__main__":
    unittest.main()
