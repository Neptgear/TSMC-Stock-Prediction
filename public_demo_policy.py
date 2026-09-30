"""A CPU-budgeted public demo policy, separate from unrestricted local research."""
from datetime import date, timedelta
import re


def public_parameters(values):
    action = values.get('do', 'view')
    if action not in {'view', 'train_transformer', 'train_tft', 'load'}:
        raise ValueError('公開 Demo 僅支援 Fetch、單一 Transformer／TFT-style 訓練及載入結果。')
    if values.get('ticker', '2330.TW') != '2330.TW':
        raise ValueError('公開 Demo 固定使用台積電 2330.TW。')
    horizon = int(values.get('horizon', 1))
    epochs = int(values.get('epochs', 3))
    if not 1 <= horizon <= 4 or not 1 <= epochs <= 5:
        raise ValueError('公開 Demo 的 Horizon 為 1～4；Epochs 為 1～5。')
    end = date.fromisoformat(values.get('end') or date.today().isoformat())
    start = date.fromisoformat(values.get('start') or (end - timedelta(days=730)).isoformat())
    if end > date.today() or not 365 <= (end - start).days <= 1096:
        raise ValueError('資料期間需為過去 1～3 年，不接受未來日期。')
    run_id = values.get('run_id', '')
    if run_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', run_id):
        raise ValueError('Run ID 格式不正確。')
    split_date = values.get('split_date', '')
    if split_date:
        split = date.fromisoformat(split_date)
        if not start < split < end:
            raise ValueError('Split Date 需位於資料期間內。')
    # Ignore advanced inputs instead of allowing arbitrary model sizes/loops.
    return dict(ticker='2330.TW', start=start.isoformat(), end=end.isoformat(),
                horizon=str(horizon), epochs=str(epochs), do=action, run_id=run_id,
                split_date=split_date, preset='custom', window='30', batch_size='64',
                lr='0.0002', d_model='32', nhead='4', num_layers='1', dim_feedforward='64',
                dropout='0.2', patience='3', weight_decay='0.001', loss='huber',
                huber_delta='0.5', train_ratio='0.8', split_mode='ratio_time', test_days='0',
                walkforward_splits='1', walkforward_epochs='1', use_fundamentals='0',
                scale_target='1', residualize='1', residual_base='close')
