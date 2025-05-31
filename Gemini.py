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
import google.generativeai as genai

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
        self.setup_gemini()
        self.load_system_prompt()
        self.chat_history = self.load_chat_history()
        self.setup_routes()
        
    def load_environment(self):
        """載入環境變數"""
        load_dotenv()
        
        # LINE Bot 設定
        self.line_access_token = os.getenv("LINE_ACCESS_TOKEN")
        self.line_channel_secret = os.getenv("LINE_CHANNEL_SECRET")
        
        # Gemini API 設定
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        # 檔案路徑設定
        self.chat_history_file = os.getenv("CHAT_HISTORY_FILE", "chat_history.json")
        self.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
        
        # 設定參數
        self.max_history_length = int(os.getenv("MAX_HISTORY_LENGTH", "4000"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "300"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_line_message_length = int(os.getenv("MAX_LINE_MESSAGE_LENGTH", "1000"))
        
        # 驗證必要的環境變數
        required_vars = [self.line_access_token, self.line_channel_secret, self.gemini_api_key]
        missing_vars = []
        
        if not self.line_access_token: missing_vars.append("LINE_ACCESS_TOKEN")
        if not self.line_channel_secret: missing_vars.append("LINE_CHANNEL_SECRET")
        if not self.gemini_api_key: missing_vars.append("GEMINI_API_KEY")
        
        if missing_vars:
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
            
    def setup_gemini(self):
        """設定 Gemini API"""
        try:
            # 配置 Gemini API
            genai.configure(api_key=self.gemini_api_key)
            
            # 建立模型實例
            generation_config = {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
            )
            
            # 測試 Gemini API 連線
            test_response = self.model.generate_content("Hello")
            logger.info("Gemini API 連線測試成功")
            logger.info(f"Gemini API 設定完成，使用模型：{self.model_name}")
            
        except Exception as e:
            logger.error(f"Gemini API 設定失敗：{e}")
            logger.error("請確認 GEMINI_API_KEY 是否正確設定")
            raise
        
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
            # 使用 Gemini API 生成摘要
            prompt = f"{summary_prompt}\n\n{full_text}"
            response = self.model.generate_content(prompt)
            
            summary = response.text.strip() if response.text else "對話摘要生成失敗"
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
            # 準備對話內容
            conversation_context = [self.system_prompt]
            
            # 添加對話歷史
            for item in self.chat_history[:-1]:  # 排除剛剛添加的用戶訊息
                if item["role"] == "assistant":
                    conversation_context.append(f"Assistant: {item['message']}")
                elif item["role"] == "user":
                    conversation_context.append(f"User: {item['message']}")
                elif item["role"] == "system":
                    conversation_context.append(f"[系統摘要] {item['message']}")
            
            # 添加當前用戶訊息
            conversation_context.append(f"User: {user_message}")
            conversation_context.append("Assistant:")
            
            # 組合完整提示
            full_prompt = "\n\n".join(conversation_context)
            
            # 呼叫 Gemini API
            response = self.model.generate_content(full_prompt)
            
            if not response.text:
                raise ValueError("Gemini API 回應為空")
            
            ai_reply = response.text.strip()
            logger.info(f"AI 回應生成成功，長度：{len(ai_reply)}")
            return ai_reply
            
        except Exception as e:
            logger.error(f"獲取 AI 回應時發生錯誤：{e}")
            if "API_KEY" in str(e):
                return "抱歉朋友，API 金鑰有問題，請聯繫管理員檢查設定 🙏"
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                return "抱歉朋友，今天的使用額度已經用完了，請明天再試試看 😅"
            else:
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
                "api_provider": "Google Gemini",
                "model": self.model_name,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route("/health", methods=["GET"])
        def health_check():
            # 檢查 Gemini API 狀態
            gemini_status = "unknown"
            try:
                test_response = self.model.generate_content("test")
                gemini_status = "healthy" if test_response.text else "unhealthy"
            except Exception as e:
                logger.error(f"Gemini API 健康檢查失敗：{e}")
                gemini_status = "unreachable"
            
            return jsonify({
                "status": "healthy",
                "gemini_status": gemini_status,
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
                if user_message.lower() in ["/clear", "/reset", "/bye", "清除記憶", "重新開始"]:
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
API 提供商：Google Gemini

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
        logger.info(f"使用 Gemini API")
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