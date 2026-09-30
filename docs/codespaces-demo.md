# Codespaces 真實模型 Demo

[建立自己的 Codespace](https://codespaces.new/Neptgear/TSMC-Stock-Prediction)
需要 GitHub 登入，並消耗建立者的時數與儲存額度。
此入口不是給評審的已運行網站。給評審的網址為
`https://<codespace-name>-5050.app.github.dev/`，需擁有者建立、啟動並公開 5050。

## 啟動與公開

1. 選擇 main、2-core，使用倉庫內 `.devcontainer/devcontainer.json`。
2. 首次自動安裝 Python 3.12、CPU PyTorch 與依賴。
3. 自動啟動 Gunicorn `cloud_demo:app`，只有 1 worker／2 threads；不要改成多 worker，否則記憶體訓練鎖與限額不共用。
4. Ports → 5050 → Port Visibility → Public；複製公開網址。
5. 或在 Codespace 內執行 `gh codespace ports visibility 5050:public -c "$CODESPACE_NAME"`。
6. 未登入視窗開啟網頁，確認 `/health` 返回 `public_demo=true`。
7. 重啟後服務會重開，但 Public 會恢復 Private，需再次公開。

## 訪客可真正操作的部分

- Fetch：取得實際 2330.TW OHLCV、計算技術指標與顯示圖表。
- Train Transformer 或 Train TFT：真正執行 CPU 訓練、測試與價格誤差計算。
- Load：載入已有實验成果；這是重播紀錄，不是當次重新訓練。

公開 Demo 固定 2330.TW、過去 1～3 年資料、Horizon 1～4、Epochs 1～5、
window 30、d_model 32、1 layer、4 heads、技術面特徵。基本面關閉，避免
把尚未完成可得時間驗證的欄位放入公開測試。進階參數不會開放無上限運算。
Train Both 在公開模式停用；Transformer 與 TFT-style 分別執行。

一次只訓練一個模型，每分鐘一次、每個服務進程每日最多 10 次。
重啟會重置記憶體計數，不等同帳號總額度／費用的硬性上限。
訓練可能需數分鐘。外部行情服務可能限流，錯誤如實顯示，不以合成價格替代。
訓練结果保存於共用 `runs/`，其他訪客可載入，請勿提交私人資料。

少量 Epochs 是流程展示，不是 20 區間 × 3 種子的正式 benchmark，
不能宣稱此 Demo 重現了全部研究結論。TFT 為專案的 TFT-style 實作，
不是完整原始 Temporal Fusion Transformer。預測非投資建議，未證明
穩定低於最後收盤價基準。

## 故障處理與檢查

- 起動狀態：`cloud-demo.log`，重開：`bash scripts/codespaces_start.sh`。
- 400：確認日期為過去 1～3 年、Horizon／Epochs 在上限內；Run ID 不含路徑。
- 429：等候目前訓練完成／分鐘限制；每日額度達到時改看既有紀錄。
- 502／離線：擁有者是否啟動 Codespace、5050 是否公開。
- Fetch／訓練失敗：外部行情限流或資料不足，保留錯誤，不宣稱成功。

上線後仍需實測新環境安裝、外部無登入存取、真實 Fetch、兩種模型各一次
訓練和 Load。**目前程式／本機驗證不等同雲端已正式上線。**

參考：[GitHub 連接埠轉送](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace)
與 [Codespaces 安全性](https://docs.github.com/en/codespaces/reference/security-in-github-codespaces)。
