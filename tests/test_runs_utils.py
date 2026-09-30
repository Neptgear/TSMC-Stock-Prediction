import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from runs_utils import load_run_results


class LoadRunResultsTests(unittest.TestCase):
    def test_missing_future_error_artifact_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config.json").write_text(
                json.dumps({"model_type": "tft", "ticker": "2330.TW"}),
                encoding="utf-8",
            )
            np.save(run_dir / "y_true.npy", np.array([100.0, 102.0]))
            np.save(run_dir / "y_pred.npy", np.array([101.0, 103.0]))

            result = load_run_results(str(run_dir))

            self.assertIsNone(result["future_point_error"])
            self.assertEqual(result["model_name"], "TFT")


if __name__ == "__main__":
    unittest.main()

