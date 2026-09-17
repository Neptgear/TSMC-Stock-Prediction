import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "train_transformer.py").read_text(encoding="utf-8")


def _function_body(name: str, next_name: str) -> str:
    start = SOURCE.index(f"def {name}(")
    end = SOURCE.index(f"def {next_name}(", start)
    return SOURCE[start:end]


class TrainingPreprocessingTests(unittest.TestCase):
    def test_active_models_do_not_backfill_and_fit_scalers_after_split(self):
        for body in (
            _function_body("run_training_transformer_seq2seq", "run_training_tft_full"),
            _function_body("run_training_tft_full", "build_seq2seq_tensors"),
        ):
            self.assertNotIn(".bfill(", body)
            self.assertIn("compute_indicators_only", body)
            self.assertNotIn("compute_features(df, target=cfg.target", body)
            self.assertIn("purge_gap = max(0, int(cfg.horizon) - 1)", body)
            self.assertIn("train_pool_end = split_idx - purge_gap", body)
            split_position = body.index("enc_train, enc_test =")
            fit_position = body.index("SequenceStandardScaler().fit(enc_train)")
            self.assertGreater(fit_position, split_position)
            self.assertIn("enc_future = obs_scaler.transform", body)
            self.assertIn('getattr(cfg, "selection_metric", "mae")', body)
            self.assertIn("copy.deepcopy(model.state_dict())", body)
            self.assertEqual(body.count("val_losses.append"), 1)

    def test_active_models_use_configured_seed(self):
        for body in (
            _function_body("run_training_transformer_seq2seq", "run_training_tft_full"),
            _function_body("run_training_tft_full", "build_seq2seq_tensors"),
        ):
            self.assertIn("seed_everything(cfg.random_seed)", body)


if __name__ == "__main__":
    unittest.main()

