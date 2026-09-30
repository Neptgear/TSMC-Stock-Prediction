# 公開發布驗證 · 2026-09-30

## 驗證環境與結果

在乾淨工作副本建立 Python 3.12 隔離環境並安裝 `requirements.txt` 後，執行：

```powershell
python -m unittest discover -s tests -v
python scripts/verify_portfolio.py
```

結果為 **16 項測試全部通過**，作品核對工具回報 `PASS`。測試涵蓋 Flask 首頁啟動、行情欄位正規化、OHLCV 稽核、保存 Run 一致性、多步日期對齊、交易日曆、只使用截止日前資料的基準、rolling-origin 區間、種子設定、防止未來資料回填，以及缺少選配錯誤檔時仍可載入既有實驗。

`verify_portfolio.py` 核對推甄說明、實驗報告、架構圖、結果圖、必要程式與測試檔案是否存在且可讀。它不是模型重新訓練，也不替實驗數字提供新的準確率證明。

## 可重現層級

| 層級 | 目前狀態 | 證據 |
| --- | --- | --- |
| 網頁與核心程式可載入 | 已自動測試 | `tests/test_app_smoke.py` |
| 資料品質與日期對齊 | 已自動測試 | `tests/test_data_quality.py`、`tests/test_multihorizon_alignment.py` |
| 防洩漏與模型種子 | 已自動測試 | `tests/test_training_preprocessing.py` |
| 保存實驗載入的容錯 | 已自動測試 | `tests/test_runs_utils.py` |
| 20 區間 × 3 種子結果 | 已保存報告與陣列摘要 | `docs/results/rolling-models-20x3-2026-09-17.md` |
| 完整環境雜湊與模型權重重放 | 尚未完成 | 未保存可部署 checkpoint 與 scaler |
| 實際交易績效 | 不在本專案範圍 | 無下單、成本或滑價模擬 |

## 公開倉庫衛生

發布版不追蹤 Python 快取、執行日誌、虛擬環境或本機金鑰。臨時除錯與資料來源探查程式集中於 `scripts/dev/`，避免和正式驗證入口混淆。公開可見不等於具備開源授權；目前沒有授予再散布或衍生作品權利。

