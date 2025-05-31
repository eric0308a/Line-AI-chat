# 智能聊天機器人

一個基於 Flask 和 LINE Bot 的智能聊天機器人，具備友善的對話風格和完整的記憶管理功能。

## ✨ 主要特色

- **友善親和**：採用溫暖自然的對話風格
- **智能記憶**：自動管理對話歷史，避免記憶過載
- **彈性配置**：系統提示詞可外部修改
- **完整日誌**：詳細的運行記錄和錯誤追蹤
- **健康檢查**：提供系統狀態監控端點

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數

複製 `.env.example` 為 `.env` 並填入您的設定：

```bash
cp .env.example .env
```

編輯 `.env` 檔案，填入必要的 API 金鑰和設定。

### 3. 自訂系統提示詞

編輯 `system_prompt.txt` 檔案來調整機器人的個性和行為：

```
你是一個友善、溫暖且樂於助人的AI助手...
```

### 4. 啟動服務

```bash
python main.py
```

## 📁 檔案結構

```
├── ollama.py               # 本地ollama 模型調用
├── OpenRouter.py           # OpenRouter API 調用
├── Gemini.py               # Gemini API 調用
├── system_prompt.txt       # 系統提示詞（可自訂）
├── .env                    # 環境變數設定
├── .env.example           # 環境變數範例
├── requirements.txt       # Python 依賴套件
├── chat_history.json      # 對話歷史記錄
├── chatbot.log           # 應用程式日誌
└── README.md             # 使用說明
```

## ⚙️ 主要設定

### 環境變數說明

| 變數名稱 | 說明 | 必填 |
|---------|------|------|
| `LINE_ACCESS_TOKEN` | LINE Bot 存取權杖 | ✅ |
| `LINE_CHANNEL_SECRET` | LINE 頻道密鑰 | ✅ |
| `OPENROUTER_URL` | OpenRouter API 端點 | ✅ |
| `OPENROUTER_API_KEY` | OpenRouter API 金鑰 | ✅ |
| `MODEL_NAME` | 使用的 AI 模型名稱 | ✅ |
| `CHAT_HISTORY_FILE` | 對話歷史檔案路徑 | ❌ |
| `SYSTEM_PROMPT_FILE` | 系統提示詞檔案路徑 | ❌ |
| `MAX_HISTORY_LENGTH` | 對話歷史最大長度 | ❌ |
| `MAX_TOKENS` | AI 回應最大字數 | ❌ |
| `TEMPERATURE` | AI 回應創意度 | ❌ |

### 使用者指令

- **清除記憶**：`/clear`、`/reset`、`清除記憶`、`重新開始`
- **查看幫助**：`/help`、`幫助`、`說明`

## 🔧 進階功能

### 對話歷史管理

- 自動壓縮過長的對話記錄
- 保留重要資訊的摘要功能
- 時間戳記記錄每次互動

### 健康檢查端點

- `GET /`：基本狀態檢查
- `GET /health`：詳細健康狀態

### 日誌系統

- 自動記錄到 `chatbot.log`
- 包含時間戳記和錯誤詳情
- 支援控制台和檔案雙重輸出

## 🎯 自訂系統提示詞

編輯 `system_prompt.txt` 來調整機器人的：

- 語言風格和用詞習慣
- 個性特質和回應態度
- 互動方式和稱呼方式
- 專業領域和知識範圍

## 🛠️ 開發說明

### 類別結構

- `ChatBot`：主要的聊天機器人類別
  - 環境設定載入
  - LINE Bot API 初始化
  - 對話歷史管理
  - AI 回應生成
  - 訊息處理和路由

### 錯誤處理

- 完整的異常捕獲機制
- 使用者友善的錯誤訊息
- 詳細的日誌記錄

### 擴展性

程式採用模組化設計，方便添加新功能：

- 新增指令處理
- 整合其他 API 服務
- 客製化回應邏輯