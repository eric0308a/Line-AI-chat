import json
import logging
import os
from datetime import datetime
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# 日誌設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.app = Flask(__name__)
        self.load_environment()
        self.setup_line_bot()
        self.setup_gemini()
        self.load_system_prompt()
        self.setup_routes()

    def load_environment(self):
        load_dotenv()
        self.line_access_token = os.getenv("LINE_ACCESS_TOKEN")
        self.line_channel_secret = os.getenv("LINE_CHANNEL_SECRET")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.history_dir = Path("history")
        self.history_dir.mkdir(exist_ok=True)
        self.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
        self.max_history_length = int(os.getenv("MAX_HISTORY_LENGTH", "4000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))

    def setup_line_bot(self):
        self.line_bot_api = LineBotApi(self.line_access_token)
        self.handler = WebhookHandler(self.line_channel_secret)

    def setup_gemini(self):
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 300,
                "response_mime_type": "text/plain"
            }
        )

    def load_system_prompt(self):
        if Path(self.system_prompt_file).exists():
            with open(self.system_prompt_file, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = self.create_default_system_prompt()

    def create_default_system_prompt(self):
        return """你是一個友善、溫暖且樂於助人的AI助手。請使用繁體中文與使用者互動，保持簡潔、親切、同理心的語調。"""

    def load_chat_history(self, user_id):
        path = self.history_dir / f"user_{user_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_chat_history(self, user_id, history):
        path = self.history_dir / f"user_{user_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def manage_chat_history(self, user_id, history, user_msg):
        history.append({
            "role": "user",
            "message": user_msg,
            "timestamp": datetime.now().isoformat()
        })
        total_length = sum(len(msg["message"]) for msg in history)
        if total_length > self.max_history_length:
            history = history[-10:]  # 保留最後 10 則
        return history

    def get_ai_response(self, history, user_msg):
        try:
            conversation = [self.system_prompt]
            for msg in history:
                if msg["role"] == "user":
                    conversation.append(f"User: {msg['message']}")
                elif msg["role"] == "assistant":
                    conversation.append(f"Assistant: {msg['message']}")
            conversation.append(f"User: {user_msg}")
            conversation.append("Assistant:")

            prompt = "\n\n".join(conversation)
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else "抱歉，我暫時無法回應。"
        except Exception as e:
            logger.error(f"Gemini API 回應失敗：{e}")
            return "發生錯誤，請稍後再試～"

    def setup_routes(self):
        @self.app.route("/callback", methods=["POST"])
        def callback():
            signature = request.headers["X-Line-Signature"]
            body = request.get_data(as_text=True)

            try:
                self.handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400)
            return "OK"

        @self.handler.add(MessageEvent, message=TextMessage)
        def handle_message(event):
            user_id = event.source.user_id
            user_msg = event.message.text.strip()

            logger.info(f"收到來自 {user_id} 的訊息：{user_msg}")

            history = self.load_chat_history(user_id)
            history = self.manage_chat_history(user_id, history, user_msg)
            ai_reply = self.get_ai_response(history, user_msg)

            history.append({
                "role": "assistant",
                "message": ai_reply,
                "timestamp": datetime.now().isoformat()
            })
            self.save_chat_history(user_id, history)

            self.line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=ai_reply)
            )

    def run(self, host="0.0.0.0", port=5566):
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    bot = ChatBot()
    bot.run()
