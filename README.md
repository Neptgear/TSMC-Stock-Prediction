# TSMC Stock Prediction

台積電股價時間序列分析與模型比較原型。專案使用 Python、Flask、PyTorch 與 Plotly，整合行情資料、技術指標、Transformer／TFT-style 模型訓練、預測圖表及歷次實驗結果，讓使用者能從網頁設定分析條件並查看結果。

本專案定位為具網頁介面的資料分析工具與研究原型。它呈現從資料處理、模型訓練到結果展示的整合過程；目前沒有券商下單、交易執行或經成本驗證的投資績效。

## 先看圖文成果

![保存的 TFT-style 實驗曲線：真實值與預測值對照](docs/assets/04-tsmc-archived-result.png)

藍線是真實值、橘線是預測值。此圖由既有 Run 的 268 筆保存資料重繪，**不是本次重新訓練、不是網頁截圖，也不是投資績效**；前處理與切分一致性仍有待驗證。

[**初學者圖文指南：看懂曲線、參數、操作步驟與誤差指標**](docs/visual-guide.md)

## 專案可以做什麼

| 功能 | 使用方式與程式依據 |
| --- | --- |
| 行情與技術圖表 | 以 `2330.TW` 為預設標的；輸入資料期間，查看價格、成交量、RSI、MACD 等圖表。資料由 `data_fetch.py` 取得，指標由 `ta_features.py` 計算 |
| 調整模型條件 | 網頁可設定預測跨度 Horizon、輸入視窗 Window、Epochs 及進階參數 |
| 比較兩種模型 | 使用 `Train Transformer`、`Train TFT` 或 `Train Both & Compare` 進行訓練及展示。雙模型操作依序訓練，並非並行執行 |
| 查看預測與診斷 | 圖表及結果區可呈現真實值與預測值、誤差與方向指標、基準比較及學習曲線；可顯示的內容取決於該次結果實際保存的欄位 |
| 保存與讀取實驗 | `runs/` 保存設定、數值陣列與 JSON；網頁讀取既有結果時使用 `load_run_results`，不會重新訓練或重新推論 |

「程式中有此流程」與「已在新環境實測通過」是不同層級。本說明於 2026-09-09 根據程式與既有實驗檔案整理，本次未重新訓練模型或完成從零安裝聯測。

## 系統組成與製作流程

1. **資料輸入**：由網頁收集股票代號、起訖時間及分析參數，取得 OHLCV 行情。Yahoo Finance 是主要來源；部分路徑具有 Alpha Vantage 或臺灣證交所的補充處理，仍受資料可用性影響。
2. **特徵處理**：計算均線、RSI、MACD、布林通道、報酬率、波動與日曆等特徵，建立滑動視窗及預測目標。程式也包含事件倒數與基本面相關處理，但不能假設所有介面選項均在每條訓練路徑生效。
3. **模型訓練**：`run_training` 分流至 Transformer Seq2Seq 或 TFT-style 路徑。`tft_model.py` 以 PyTorch 組合變數門控、LSTM 編碼／解碼、多頭注意力及分位數輸出；屬專案中的 TFT-style 實作，尚未證明與原始 TFT 論文完全等價。
4. **結果展示**：Flask 將訓練結果交給 Jinja 模板與 Plotly，呈現預測曲線、指标與診斷資訊。
5. **實驗留存**：將設定、預測陣列、日期、指標及可用診斷資料存入 `runs/`，供後續回顧與比較。

## 專案角色與實作能力

依製作者林琨茂的專題說明，本專案由其個人發想與製作，開發過程大量使用 AI 協助程式產生、修改與文件整理。此作品用來呈現需求規劃、資料分析、模型與網頁整合的實作經驗。程式存在與實驗檔案本身不等於每個模組均獨立手寫，也不代表提出新的模型架構。

適合在專題展示時說明的內容包括：如何將股價問題拆成資料、特徵、模型和展示流程；如何設定輸入視窗與預測跨度；如何讀取歷次結果；以及如何辨認指標、資料切分和實作限制。

## 本機啟動

目前倉庫沒有鎖定版本的依賴清單。以下依程式匯入項目提供環境建立起點，尚非經新環境聯測的安裝保證；PyTorch 版本須配合 Python、作業系統與運算裝置選擇。

```powershell
git clone https://github.com/Neptgear/TSMC-Stock-Prediction.git
cd TSMC-Stock-Prediction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install flask numpy pandas torch yfinance requests
.\.venv\Scripts\python.exe -m flask --app app:create_app run --host 127.0.0.1 --port 5000
```

在瀏覽器開啟 `http://127.0.0.1:5000`。私人倉庫需有 GitHub 存取權限才能複製。網頁圖表若使用外部 Plotly 資源，還需要對應的網路連線。

Alpha Vantage 為選用資料來源，可使用 `ALPHAVANTAGE_API_KEY` 環境變數，或參考 `secrets_local.example.py` 建立未提交的本機設定；不要把實際金鑰貼入說明或提交到版本控制。

### 操作順序

1. 先以 `Fetch` 確認股票與資料期間能取得資料。
2. 選擇 Horizon、Window、Epochs，使用 Custom 時確認進階參數；選用 Preset 會覆寫部分設定。
3. 執行單一模型或 `Train Both & Compare`；訓練在請求中進行，CPU 執行可能需要等待。
4. 查看結果與診斷，使用介面的儲存功能留下本次實驗。
5. 從既有 Run 選擇紀錄後讀取，這一步展示已保存的結果，不是對當下股價重新預測。頁面其他行情流程仍可能存取網路，不能因此宣稱整個網頁可離線使用。

命令列也可啟動訓練，例如：

```powershell
python train_transformer.py --ticker 2330.TW --start 2020-01-01 --horizon 5 --window 90 --epochs 10 --model_type tft
```

## 如何閱讀已保存的成果

以 [`saved_run_20251203_155344_tft`](runs/saved_run_20251203_155344_tft) 為例，倉庫保存 268 筆真實值與預測值。根據 `test_dates.json`，圖表日期為 2024-10-22 至 2025-11-26。由 `y_true.npy`、`y_pred.npy` 重新計算的 RMSE 約 29.891、MAE 約 21.633，與該資料夾 `metrics.json` 的對應數值一致；這只是對歷史數值檔案的核算，不是重新訓練或外部樣本驗證。

| 檔案 | 內容 |
| --- | --- |
| `config.json` | 當次儲存的設定資訊 |
| `y_true.npy`、`y_pred.npy` | 已保存的真實值與預測值，可用 NumPy 讀取 |
| `test_dates.json` | 上述結果使用的日期標籤 |
| `metrics.json` | 誤差與方向分類指標；不是交易收益 |
| `split_info.json` | 訓練、驗證與測試區間的紀錄 |
| `diagnostics.json` | 部分實驗保存基準、學習曲線與其他診斷 |

既有 Run 不應一律視為完整模型檢查點；目前所核對的倉庫樹沒有 `model.pt` 與 scaler 檔案。讀取保存曲線可使用陣列，但不能假定可用這些 Run 對新資料重新推論。

## 已知限制與後續改進

- **資料前處理需要改善**：目前兩條主要訓練路徑在切分前，對完整特徵資料計算標準化統計量，並包含 `bfill`；因此存在未來資訊進入前處理的風險。後續應以訓練區間擬合前處理，並檢查每項特徵在當時是否可取得。
- **保存設定的一致性尚待釐清**：上述 Run 的 `config.json` 記錄 `train_ratio=0.6`，`split_info.json` 卻記錄 `ratio_samples_0.80`；後者的測試日期範圍也與 `test_dates.json` 不同，可能涉及視窗錨點與目標日期等語意，需追查後再定論。故不能只引用漂亮指標宣稱泛化能力。
- **部分進階參數需逐路徑確認**：介面存在 Walk Splits、Split Mode 與基本面選項，但主要 Seq2Seq／TFT 分流並不等同於後面的舊訓練路徑；不能將所有選項標為已驗證有效。
- **模型屬研究實作**：TFT-style 與 Transformer 需要進一步做基準、消融及獨立時段測試；預測曲線與歷史方向指標不構成交易獲利證明。
- **工程可重現性尚待補齊**：後續需鎖定依賴、補環境與流程測試、核對資料來源，並整理已追蹤的快取與日誌。倉庫已包含 `__pycache__/secrets_local.cpython-313.pyc`，公開前應檢查是否含本機設定；本次未讀取其內容，也未刪除或改寫歷史。

## 主要檔案

| 路徑 | 職責 |
| --- | --- |
| [`app.py`](app.py) | Flask 入口、介面參數、模型呼叫、實驗保存 |
| [`templates/index.html`](templates/index.html) | 參數表單、結果區及 Plotly 圖表 |
| [`data_fetch.py`](data_fetch.py) | 行情、基本面與事件資料處理 |
| [`ta_features.py`](ta_features.py) | 技術特徵與時間序列視窗工具 |
| [`train_transformer.py`](train_transformer.py) | 訓練分流、模型、指標及診斷 |
| [`tft_model.py`](tft_model.py) | TFT-style PyTorch 模型組件 |
| [`runs_utils.py`](runs_utils.py) | Run 清單、歷史結果讀取及相關工具 |
| [`runs/`](runs/) | 已保存的實驗資料 |

說明核對基準：原始碼提交 `77bf92e434a7b1a3c8347cb9990a9cd2843d39d9`，2026-09-09。此 README 更新只補充說明，沒有修改模型或資料處理邏輯。