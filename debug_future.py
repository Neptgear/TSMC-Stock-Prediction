from train_transformer import TrainConfig, run_training


def main() -> None:
    base_cfg = dict(
        ticker="2330.TW",
        start="2010-01-01",
        end=None,
        target="next_close",
        horizon=1,
        window_size=60,
        epochs=1,
        use_fundamentals=True,
    )

    for model_type in ("transformer", "tft"):
        print(f"=== {model_type} ===")
        cfg = TrainConfig(model_type=model_type, **base_cfg)
        res = run_training(cfg)
        fp = res.get("future_point")
        print("future_point:", fp)


if __name__ == "__main__":
    main()

