import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        """初始化聊天機器人"""
        self.app = Flask(__name__)
        self.load_environment()
        self.setup_line_bot()
        self.setup_ollama()
        self.load_system_prompt()
        self.chat_history = self.load_chat_history()
        self.setup_routes()
        
    def load_environment(self):
        """載入環境變數"""
        load_dotenv()
        
        # LINE Bot 設定
        self.line_access_token = os.getenv("LINE_ACCESS_TOKEN")
        self.line_channel_secret = os.getenv("LINE_CHANNEL_SECRET")
        
        # Ollama API 設定
        self.ollama_url = "http://localhost:11434"
        self.model_name = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        
        # 檔案路徑設定
        self.chat_history_file = os.getenv("CHAT_HISTORY_FILE", "chat_history.json")
        self.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
        
        # 設定參數
        self.max_history_length = int(os.getenv("MAX_HISTORY_LENGTH", "4000"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "300"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_line_message_length = int(os.getenv("MAX_LINE_MESSAGE_LENGTH", "1000"))
        
        # 驗證必要的環境變數
        required_vars = [self.line_access_token, self.line_channel_secret]
        
        if not all(required_vars):
            missing_vars = []
            if not self.line_access_token: missing_vars.append("LINE_ACCESS_TOKEN")
            if not self.line_channel_secret: missing_vars.append("LINE_CHANNEL_SECRET")
            
            raise ValueError(f"請設定以下環境變數：{', '.join(missing_vars)}")
        
        logger.info("環境變數載入完成")
        
    def setup_line_bot(self):
        """設定 LINE Bot API"""
        try:
            self.line_bot_api = LineBotApi(self.line_access_token)
            self.handler = WebhookHandler(self.line_channel_secret)
            logger.info("LINE Bot API 設定完成")
        except Exception as e:
            logger.error(f"LINE Bot API 設定失敗：{e}")
            raise
            
    def setup_ollama(self):
        """設定 Ollama API"""
        self.ollama_chat_url = f"{self.ollama_url}/api/chat"
        self.ollama_generate_url = f"{self.ollama_url}/api/generate"
        
        # 測試 Ollama 連線
        try:
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            test_response.raise_for_status()
            logger.info("Ollama API 連線測試成功")
        except Exception as e:
            logger.warning(f"Ollama API 連線測試失敗：{e}")
            logger.warning("請確認 Ollama 已經啟動並運行在指定位址")
        
        logger.info(f"Ollama API 設定完成，使用模型：{self.model_name}")
        
    def load_system_prompt(self):
        """載入系統提示詞"""
        try:
            prompt_path = Path(self.system_prompt_file)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as file:
                    self.system_prompt = file.read().strip()
                logger.info(f"系統提示詞從 {self.system_prompt_file} 載入完成")
            else:
                # 如果檔案不存在，建立預設的系統提示詞檔案
                self.system_prompt = self.create_default_system_prompt()
                self.save_system_prompt()
                logger.info(f"建立預設系統提示詞檔案：{self.system_prompt_file}")
        except Exception as e:
            logger.error(f"載入系統提示詞失敗：{e}")
            self.system_prompt = self.create_default_system_prompt()
            
    def create_default_system_prompt(self):
        """建立預設的系統提示詞"""
        return """你是一個友善、溫暖且樂於助人的AI助手。請遵循以下特點：

1. 語言風格：
   - 全程使用繁體中文回應
   - 使用親切自然的聊天語調
   - 每次回答簡潔明瞭，不超過200字
   - 如需長篇說明，可分多次回覆

2. 個性特質：
   - 保持積極正面的態度
   - 展現同理心和理解力
   - 主動關心使用者的需求
   - 提供實用的建議和協助

3. 互動方式：
   - 稱呼使用者為「朋友」
   - 自稱為「小助手」或「我」
   - 適時使用表情符號增加親和力
   - 遇到不理解的問題時，禮貌地請求澄清

4. 回應原則：
   - 直接提供對話內容，不加系統標籤
   - 保持專業但不失溫度
   - 尊重使用者的隱私和感受
   - 在能力範圍內盡力協助

請用這種風格與使用者互動，讓對話既有幫助又充滿溫暖。"""

    def save_system_prompt(self):
        """儲存系統提示詞到檔案"""
        try:
            with open(self.system_prompt_file, "w", encoding="utf-8") as file:
                file.write(self.system_prompt)
            logger.info(f"系統提示詞已儲存至 {self.system_prompt_file}")
        except Exception as e:
            logger.error(f"儲存系統提示詞失敗：{e}")

    def load_chat_history(self):
        """載入對話歷史"""
        try:
            with open(self.chat_history_file, "r", encoding="utf-8") as file:
                history = json.load(file)
                logger.info(f"載入 {len(history)} 條對話記錄")
                return history
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.info("未找到對話歷史檔案，建立新的對話記錄")
            return []
        except Exception as e:
            logger.error(f"載入對話歷史失敗：{e}")
            return []

    def save_chat_history(self):
        """儲存對話歷史"""
        try:
            with open(self.chat_history_file, "w", encoding="utf-8") as file:
                json.dump(self.chat_history, file, ensure_ascii=False, indent=2)
            logger.info(f"對話歷史已儲存，共 {len(self.chat_history)} 條記錄")
        except Exception as e:
            logger.error(f"儲存對話歷史失敗：{e}")

    def get_conversation_summary(self, conversation):
        """生成對話摘要"""
        summary_prompt = """請閱讀以下對話記錄，並用繁體中文簡潔摘要重點內容。

摘要格式：
- 主要話題：[用戶關心的核心問題]
- 重要信息：[對話中的關鍵信息點]
- 用戶需求：[用戶希望得到的幫助]
- 待解決問題：[如果有未完成的討論]

請保持摘要簡潔，重點突出，方便後續對話參考。"""

        full_text = "\n".join([f"{item['role']}: {item['message']}" for item in conversation])
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"{summary_prompt}\n\n{full_text}",
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_ctx": 4096
                }
            }
            
            response = requests.post(
                self.ollama_generate_url, 
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            summary = result.get("response", "").strip()
            logger.info("對話摘要生成成功")
            return summary
            
        except Exception as e:
            logger.error(f"生成對話摘要失敗：{e}")
            return "對話摘要：先前的對話內容因長度限制已被壓縮。"

    def manage_chat_history(self, user_message):
        """管理對話歷史長度"""
        # 添加用戶訊息
        self.chat_history.append({
            "role": "user", 
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # 檢查總長度
        total_length = sum(len(item["message"]) for item in self.chat_history)
        
        if total_length > self.max_history_length:
            logger.info(f"對話歷史超過限制 ({total_length} > {self.max_history_length})，開始壓縮")
            
            # 保留最近的一些對話，其餘進行摘要
            recent_count = len(self.chat_history) // 3  # 保留最近1/3的對話
            recent_messages = self.chat_history[-recent_count:]
            old_messages = self.chat_history[:-recent_count]
            
            if old_messages:
                summary = self.get_conversation_summary(old_messages)
                self.chat_history = [
                    {
                        "role": "system", 
                        "message": f"對話摘要：{summary}",
                        "timestamp": datetime.now().isoformat()
                    }
                ] + recent_messages
                
                logger.info("對話歷史壓縮完成")

    def get_ai_response(self, user_message):
        """獲取 AI 回應"""
        try:
            # 準備對話內容 - 使用 Ollama 的 chat 格式
            messages = [{"role": "system", "content": self.system_prompt}]
            
            for item in self.chat_history:
                if item["role"] == "assistant":
                    messages.append({"role": "assistant", "content": item["message"]})
                elif item["role"] == "user":
                    messages.append({"role": "user", "content": item["message"]})
                elif item["role"] == "system":
                    # 將系統摘要作為 user 訊息處理
                    messages.append({"role": "user", "content": f"[系統摘要] {item['message']}"})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": 4096,
                    "num_predict": self.max_tokens
                }
            }

            response = requests.post(
                self.ollama_chat_url, 
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            ai_reply = result.get("message", {}).get("content", "").strip()
            
            if not ai_reply:
                raise ValueError("AI 回應為空")
                
            logger.info(f"AI 回應生成成功，長度：{len(ai_reply)}")
            return ai_reply
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 請求失敗：{e}")
            return "抱歉朋友，我現在遇到了一些技術問題，請稍後再試試看 🙏\n（請確認 Ollama 服務正在運行）"
        except Exception as e:
            logger.error(f"獲取 AI 回應時發生錯誤：{e}")
            return "不好意思，我剛才有點恍神，可以請你再說一次嗎？ 😅"

    def split_message(self, message):
        """分割長訊息"""
        messages = []
        while message:
            if len(message) <= self.max_line_message_length:
                messages.append(TextSendMessage(text=message))
                break
            else:
                # 尋找適當的分割點（避免在句子中間分割）
                split_point = self.max_line_message_length
                for i in range(self.max_line_message_length - 50, self.max_line_message_length):
                    if message[i] in '。！？\n':
                        split_point = i + 1
                        break
                
                messages.append(TextSendMessage(text=message[:split_point]))
                message = message[split_point:]
        
        return messages

    def setup_routes(self):
        """設定路由"""
        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({
                "status": "運行中",
                "message": "聊天機器人正常運作",
                "ollama_url": self.ollama_url,
                "model": self.model_name,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route("/health", methods=["GET"])
        def health_check():
            # 檢查 Ollama 服務狀態
            ollama_status = "unknown"
            try:
                test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                ollama_status = "healthy" if test_response.status_code == 200 else "unhealthy"
            except:
                ollama_status = "unreachable"
            
            return jsonify({
                "status": "healthy",
                "ollama_status": ollama_status,
                "chat_history_count": len(self.chat_history),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route("/callback", methods=["POST"])
        def callback():
            signature = request.headers.get("X-Line-Signature", "")
            body = request.get_data(as_text=True)

            try:
                self.handler.handle(body, signature)
            except InvalidSignatureError:
                logger.error("LINE Webhook 簽名驗證失敗")
                return "Bad Request", 400
            except Exception as e:
                logger.error(f"處理 webhook 時發生錯誤：{e}")
                return "Internal Server Error", 500

            return "OK"

        @self.handler.add(MessageEvent, message=TextMessage)
        def handle_message(event):
            try:
                user_message = event.message.text.strip()
                user_id = event.source.user_id
                
                logger.info(f"收到用戶訊息 ({user_id[:8]}...)：{user_message[:50]}...")

                # 處理特殊指令
                if user_message.lower() in ["/clear", "/reset","/bye" , "清除記憶", "重新開始"]:
                    self.chat_history = []
                    self.save_chat_history()
                    reply_text = "好的朋友！我們的對話記錄已經清除，可以重新開始聊天了 ✨"
                    self.line_bot_api.reply_message(
                        event.reply_token, 
                        TextSendMessage(text=reply_text)
                    )
                    logger.info(f"用戶 {user_id[:8]}... 清除對話記錄")
                    return

                if user_message.lower() in ["/help", "幫助", "說明"]:
                    help_text = f"""嗨朋友！我是你的AI小助手 🤖

當前使用模型：{self.model_name}

我可以幫你：
• 回答各種問題
• 提供建議和協助
• 進行日常聊天

特殊指令：
• 發送「清除記憶」重新開始對話
• 發送「幫助」查看此說明

有什麼想聊的嗎？我隨時都在！ 😊"""
                    
                    self.line_bot_api.reply_message(
                        event.reply_token, 
                        TextSendMessage(text=help_text)
                    )
                    return

                # 管理對話歷史
                self.manage_chat_history(user_message)

                # 獲取 AI 回應
                ai_reply = self.get_ai_response(user_message)

                # 記錄 AI 回應
                self.chat_history.append({
                    "role": "assistant", 
                    "message": ai_reply,
                    "timestamp": datetime.now().isoformat()
                })
                self.save_chat_history()

                # 分割並發送訊息
                reply_messages = self.split_message(ai_reply)
                self.line_bot_api.reply_message(event.reply_token, reply_messages)
                
                logger.info(f"成功回應用戶 {user_id[:8]}...")

            except LineBotApiError as e:
                logger.error(f"LINE Bot API 錯誤：{e}")
            except Exception as e:
                logger.error(f"處理訊息時發生錯誤：{e}")
                try:
                    error_reply = "抱歉朋友，我遇到了一些問題，請稍後再試試看 🙏"
                    self.line_bot_api.reply_message(
                        event.reply_token, 
                        TextSendMessage(text=error_reply)
                    )
                except:
                    pass

    def run(self, host="0.0.0.0", port=5566, debug=False):
        """運行應用程式"""
        logger.info(f"聊天機器人啟動，監聽 {host}:{port}")
        logger.info(f"使用 Ollama 服務：{self.ollama_url}")
        logger.info(f"使用模型：{self.model_name}")
        self.app.run(host=host, port=port, debug=debug)

# 建立並運行聊天機器人
if __name__ == "__main__":
    try:
        chatbot = ChatBot()
        chatbot.run()
    except Exception as e:
        logger.error(f"啟動聊天機器人失敗：{e}")
        raise