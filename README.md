# AI 聊天機器人 🤖

這是一個基於 **LINE Messaging API** 和 **Google Gemini AI** 所開發的智慧聊天機器人。它能夠與用戶進行多輪對話，並支援個人化的系統提示詞設定與聊天紀錄管理功能。


## ✨ 功能特色

* **智能對話**: 整合 Google Gemini AI 模型，提供流暢自然的對話體驗。
* **個人化提示詞**: 允許用戶自訂 AI 的行為模式（系統提示詞），讓機器人更符合個人需求。
* **聊天紀錄管理**: 自動管理對話歷史，並提供清除歷史紀錄的功能。
* **易於部署**: 透過環境變數輕鬆設定各項參數。
* **繁體中文支援**: 預設使用繁體中文進行互動。


## 🛠️ 環境設定

### 前置準備

在執行此專案之前，請確保您已完成以下準備：

1.  **Python 環境**: 建議使用 Python 3.9 或更高版本。
2.  **LINE Developers 帳號**:
    * 建立一個 LINE 官方帳號。
    * 在 Messaging API 設定中，取得您的 **Channel Access Token** 和 **Channel Secret**。
    * 設定 Webhook URL 為您的伺服器位址 (例如: `https://your-domain.com/callback`)。
3.  **Google Cloud 帳號**:
    * 啟用 Gemini API。
    * 建立一個 API 金鑰。

### 安裝依賴套件

請使用 `pip` 安裝專案所需的 Python 函式庫：

```bash
pip install -r requirements.txt
```

如果您沒有 requirements.txt 檔案，您可以手動建立它，並包含以下內容：

```
Flask
line-bot-sdk
python-dotenv
google-generativeai
```

### 環境變數設定

請在專案根目錄建立一個 .env 檔案，並填入以下資訊：

```
LINE_ACCESS_TOKEN="您的 LINE Channel Access Token"
LINE_CHANNEL_SECRET="您的 LINE Channel Secret"
GEMINI_API_KEY="您的 Google Gemini API 金鑰"
GEMINI_MODEL="gemini-1.5-flash" # 可選，預設為 gemini-1.5-flash
SYSTEM_PROMPT_FILE="system_prompt.txt" # 可選，預設為 system_prompt.txt
MAX_HISTORY_LENGTH="4000" # 可選，預設為 4000 個字元
TEMPERATURE="0.7" # 可選，預設為 0.7
```

### 預設系統提示詞

您可以在專案根目錄下建立一個名為 system_prompt.txt 的檔案，用來自訂 AI 的預設行為。如果此檔案不存在，機器人會使用以下預設提示詞：

```
你是一個友善、溫暖且樂於助人的AI助手。請使用繁體中文與使用者互動，保持簡潔、親切、同理心的語調。
```

## 🚀 執行專案

在完成環境設定後，您可以執行 main.py 啟動聊天機器人：

```bash
python main.py
```

預設情況下，應用程式將會運行在 http://0.0.0.0:5566。請確保您的 LINE Webhook URL 配置與此相符。

## 💬 使用方式

透過 LINE 與機器人互動，除了日常對話外，您還可以使用以下指令來管理機器人：

- 設定提示詞: 輸入 設定提示詞。機器人會顯示目前的提示詞，並提示您輸入新的提示詞。輸入新的內容後，機器人會將其儲存並應用。
- 清除提示詞: 輸入 清除提示詞。這將會刪除您個人設定的提示詞，並恢復為預設的系統提示詞。
- 清除聊天紀錄: 輸入 /bye。這將會清空您與機器人之間的所有聊天紀錄，從此開始一段新的對話。

## 📁 專案結構

```
.
├── app.py                 # 專案主程式
├── .env                    # 環境變數設定檔 (請自行建立)
├── system_prompt.txt       # 預設系統提示詞檔案 (可選)
├── requirements.txt        # Python 依賴套件列表 (請自行建立)
├── history/                # 儲存用戶聊天紀錄的資料夾
│   └── user_xxxxxxxx.json  # 特定用戶的聊天紀錄檔案
└── prompts/                # 儲存用戶自訂提示詞的資料夾
    └── user_xxxxxxxx.txt   # 特定用戶的自訂提示詞檔案
```

## ⚠️ 注意事項

- 請確保您的 LINE Webhook URL 可以被外部網路訪問。在本地開發時，您可能需要使用如 ngrok 等工具將本地伺服器暴露到公網。
- 聊天紀錄和個人提示詞會儲存在 history/ 和 prompts/ 資料夾中。請勿手動修改這些檔案，除非您明確知道自己在做什麼。
- 當聊天紀錄長度超過 MAX_HISTORY_LENGTH 時，機器人會保留最新的 10 則訊息以維持對話的連貫性。
- TEMPERATURE 參數會影響 AI 回應的隨機性。較高的值會產生更多樣但可能不那麼精確的回應，較低的值會更專注和保守。您可以根據需求調整此值。