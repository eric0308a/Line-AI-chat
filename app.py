import json
import logging
import os
from datetime import datetime
from flask import Flask, request, abort
import threading
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(self):
        self.app = Flask(__name__)
        self.load_environment()
        self.setup_line_bot()
        self.setup_gemini()
        self.setup_routes()

    def load_environment(self):
        load_dotenv()
        self.line_access_token = os.getenv("LINE_ACCESS_TOKEN")
        self.line_channel_secret = os.getenv("LINE_CHANNEL_SECRET")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        self.history_dir = Path("history")
        self.history_dir.mkdir(exist_ok=True)
        self.image_dir = Path("images")
        self.image_dir.mkdir(exist_ok=True)
        
        self.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
        self.max_history_tokens = int(os.getenv("MAX_HISTORY_TOKENS", 8000))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        self.prompts_dir = Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)

    def setup_line_bot(self):
        self.configuration = Configuration(access_token=self.line_access_token)
        self.handler = WebhookHandler(self.line_channel_secret)
        self.messaging_api = MessagingApi(ApiClient(self.configuration))
        self.messaging_api_blob = MessagingApiBlob(ApiClient(self.configuration))

    def setup_gemini(self):
        genai.configure(api_key=self.gemini_api_key)
        self.generation_config_dict = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config_dict
        )

    def get_system_prompt(self, user_id):
        user_prompt_file = self.prompts_dir / f"user_{user_id}.txt"
        if user_prompt_file.exists():
            return user_prompt_file.read_text(encoding="utf-8").strip()
        if Path(self.system_prompt_file).exists():
            with open(self.system_prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "你是一個友善、溫暖且樂於助人的AI助手。請使用繁體中文與使用者互動，保持簡潔、親切、同理心的語調。"

    def load_chat_history(self, user_id):
        path = self.history_dir / f"user_{user_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def save_chat_history(self, user_id, history):
        path = self.history_dir / f"user_{user_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def manage_chat_history(self, history):
        total_tokens = sum(len(str(part)) * 2 for turn in history for part in turn.get('parts', []))
        while total_tokens > self.max_history_tokens and len(history) > 1:
            removed_turn = history.pop(0)
            total_tokens -= sum(len(str(part)) * 2 for part in removed_turn.get('parts', []))
            for part in removed_turn.get('parts', []):
                if isinstance(part, str) and part.startswith(str(self.image_dir)):
                    try: Path(part).unlink()
                    except OSError: pass
        return history
    
    def get_ai_response(self, user_id, history, user_content):
        try:
            system_prompt = self.get_system_prompt(user_id)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config_dict,
                system_instruction=system_prompt,
            )
            gemini_history = []
            for turn in history:
                role = "model" if turn["role"] == "assistant" else turn["role"]
                if 'message' in turn and 'parts' not in turn:
                    turn['parts'] = [turn['message']]
                parts = []
                for part in turn.get('parts', []):
                    if isinstance(part, str) and part.startswith(str(self.image_dir)):
                        try: parts.append(Image.open(part))
                        except FileNotFoundError: parts.append("(圖片已遺失)")
                    else:
                        parts.append(part)
                gemini_history.append({'role': role, 'parts': parts})
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_content)
            return response.text.strip() if response.text else "抱歉，我暫時無法回應。"
        except Exception as e:
            logger.error(f"Gemini API 回應失敗：{e}", exc_info=True)
            return "發生錯誤，請稍後再試～"

    def setup_routes(self):
        
        def background_task(user_id, event_type, data):
            try:
                if event_type == 'text':
                    user_content = [data]
                    storable_parts = user_content
                elif event_type == 'image':
                    image_path = self.image_dir / f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                    image_obj = Image.open(io.BytesIO(data))
                    image_obj.save(image_path, 'PNG')
                    user_content = ["請詳細描述這張圖片的內容。如果圖片中有文字，也請一併列出。", image_obj]
                    storable_parts = [user_content[0], str(image_path)]
                else:
                    return

                history = self.load_chat_history(user_id)
                ai_reply = self.get_ai_response(user_id, history, user_content)

                history.append({"role": "user", "parts": storable_parts})
                history.append({"role": "assistant", "parts": [ai_reply]})
                history = self.manage_chat_history(history)
                self.save_chat_history(user_id, history)

                self.messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=ai_reply)]
                    )
                )

            except Exception as e:
                logger.error(f"背景任務失敗: {e}", exc_info=True)
                self.messaging_api.push_message(
                    to=user_id,
                    messages=[TextMessage(text="抱歉，處理您的請求時發生了一點問題。")]
                )

        @self.app.route("/callback", methods=["POST"])
        def callback():
            signature = request.headers["X-Line-Signature"]
            body = request.get_data(as_text=True)
            try:
                self.handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400)
            return "OK"

        @self.handler.add(MessageEvent, message=TextMessageContent)
        def handle_text_message(event):
            user_id = event.source.user_id
            user_msg = event.message.text.strip()
            
            def reply_sync(text):
                from linebot.v3.messaging import ReplyMessageRequest
                self.messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=text)]
                    )
                )

            prompt_flag_file = self.prompts_dir / f"user_{user_id}_awaiting.txt"
            if prompt_flag_file.exists():
                user_prompt_file = self.prompts_dir / f"user_{user_id}.txt"
                user_prompt_file.write_text(user_msg, encoding="utf-8")
                prompt_flag_file.unlink()
                reply_sync("✅ 系統提示詞已更新！")
                return

            if user_msg == "設定提示詞":
                current_prompt = self.get_system_prompt(user_id)
                prompt_flag_file.write_text("awaiting", encoding="utf-8")
                reply_sync(f"🔧 現在的提示詞如下：\n\n{current_prompt}\n\n請輸入你想要變更的新提示詞：")
                return

            if user_msg == "清除提示詞":
                prompt_path = self.prompts_dir / f"user_{user_id}.txt"
                if prompt_path.exists():
                    prompt_path.unlink()
                    reply_sync("✅ 已清除使用者提示詞，恢復為預設提示詞。")
                else:
                    reply_sync("ℹ️ 你尚未自訂提示詞，已使用預設提示詞。")
                return

            if user_msg == "/bye":
                history_path = self.history_dir / f"user_{user_id}.json"
                if history_path.exists():
                    history_path.unlink()
                reply_sync("🗑️ 已清除你的聊天紀錄，從頭開始囉！")
                return

            logger.info(f"收到來自 {user_id} 的文字訊息，將啟動背景AI處理：{user_msg}")
            threading.Thread(target=background_task, args=(user_id, 'text', user_msg)).start()

        @self.handler.add(MessageEvent, message=ImageMessageContent)
        def handle_image_message(event):
            logger.info(f"收到來自 {event.source.user_id} 的圖片訊息，將啟動背景AI處理")
            try:
                message_content = self.messaging_api_blob.get_message_content(message_id=event.message.id)
                threading.Thread(target=background_task, args=(event.source.user_id, 'image', message_content)).start()
            except Exception as e:
                logger.error(f"處理圖片訊息時下載失敗: {e}", exc_info=True)

    def run(self, host="0.0.0.0", port=5566):
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    bot = ChatBot()
    port = int(os.environ.get("PORT", 5566))
    bot.run(host="0.0.0.0", port=port)