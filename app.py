import json
import logging
import os
from datetime import datetime
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
# ⭐️ 1. 匯入 ImageMessage 來處理圖片訊息
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
# ⭐️ 2. 匯入處理圖片所需的函式庫
from PIL import Image
import io

# 日誌設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.app = Flask(__name__)
        self.load_environment()
        self.setup_line_bot()
        self.setup_gemini()
        # self.load_system_prompt() # ⭐️ 重新組織了 prompt 的載入方式
        self.setup_routes()

    def load_environment(self):
        load_dotenv()
        self.line_access_token = os.getenv("LINE_ACCESS_TOKEN")
        self.line_channel_secret = os.getenv("LINE_CHANNEL_SECRET")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # 確保使用支援視覺的模型，gemini-1.5-flash 非常適合
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        self.history_dir = Path("history")
        self.history_dir.mkdir(exist_ok=True)
        # ⭐️ 3. 新增一個資料夾來存放使用者上傳的圖片
        self.image_dir = Path("images")
        self.image_dir.mkdir(exist_ok=True)
        
        self.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
        self.max_history_tokens = int(os.getenv("MAX_HISTORY_TOKENS", 8000)) # ⭐️ 建議用 token 數來管理歷史紀錄長度
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        self.prompts_dir = Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)


    def setup_line_bot(self):
        self.line_bot_api = LineBotApi(self.line_access_token)
        self.handler = WebhookHandler(self.line_channel_secret)

    def setup_gemini(self):
        genai.configure(api_key=self.gemini_api_key)
        # ⭐️ 4. 修改 generation_config，移除 response_mime_type，讓模型能更有彈性地回應
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192, # 增加輸出的 token 上限
            },
            # safety_settings=... # 可選：設定安全過濾等級
        )

    def get_system_prompt(self, user_id):
        # 優先讀取使用者自定義的提示詞
        user_prompt_file = self.prompts_dir / f"user_{user_id}.txt"
        if user_prompt_file.exists():
            return user_prompt_file.read_text(encoding="utf-8").strip()
        
        # 若無，則讀取全域的系統提示詞
        if Path(self.system_prompt_file).exists():
            with open(self.system_prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        # 最後使用預設提示詞
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
        # ⭐️ 5. 修改歷史紀錄管理方式，以 token 數為基礎
        # 簡易估算 token (1 個中文字約 2-3 token，此處簡化為 2)
        total_tokens = sum(len(str(part)) * 2 for turn in history for part in turn.get('parts', []))
        
        while total_tokens > self.max_history_tokens and len(history) > 1:
            removed_turn = history.pop(0) # 從最舊的開始移除
            total_tokens -= sum(len(str(part)) * 2 for part in removed_turn.get('parts', []))
            
            # 如果移除的是 user 的圖片，順便刪除本地檔案
            for part in removed_turn.get('parts', []):
                if isinstance(part, str) and part.startswith(str(self.image_dir)):
                    try:
                        Path(part).unlink()
                        logger.info(f"已刪除過期圖片: {part}")
                    except OSError as e:
                        logger.error(f"刪除圖片失敗: {e}")
        return history

    def get_ai_response(self, user_id, user_content):
        # ⭐️ 6. 全面重構 AI 回應函式，使其支援多模態輸入
        try:
            system_prompt = self.get_system_prompt(user_id)
            history = self.load_chat_history(user_id)

            # 將歷史紀錄轉換為 Gemini API 接受的格式
            gemini_history = []
            for turn in history:
                gemini_turn = {"role": turn["role"], "parts": []}
                for part in turn["parts"]:
                    # 如果 part 是圖片路徑，就讀取圖片
                    if isinstance(part, str) and part.startswith(str(self.image_dir)):
                        try:
                            img = Image.open(part)
                            gemini_turn["parts"].append(img)
                        except FileNotFoundError:
                            logger.warning(f"找不到歷史圖片檔案: {part}，將略過此部分。")
                            gemini_turn["parts"].append("(圖片已遺失)")
                    else: # 否則就是文字
                        gemini_turn["parts"].append(part)
                gemini_history.append(gemini_turn)

            # 建立 Gemini 的對話 session
            chat_session = self.model.start_chat(history=gemini_history)

            # 傳送新的使用者訊息 (可以是文字或圖片)
            response = chat_session.send_message(user_content)
            ai_reply = response.text.strip()

            # 更新並儲存歷史紀錄
            history.append({"role": "user", "parts": user_content})
            history.append({"role": "model", "parts": [ai_reply]})
            history = self.manage_chat_history(history) # 管理歷史紀錄長度
            self.save_chat_history(user_id, history)

            return ai_reply if ai_reply else "抱歉，我暫時無法回應。"
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

        # ⭐️ 7. 修改文字訊息處理器，使其與新的架構相容
        @self.handler.add(MessageEvent, message=TextMessage)
        def handle_text_message(event):
            user_id = event.source.user_id
            user_msg = event.message.text.strip()
            logger.info(f"收到來自 {user_id} 的文字訊息：{user_msg}")

            prompt_flag_file = self.prompts_dir / f"user_{user_id}_awaiting.txt"

            # 功能 1️⃣：設定提示詞
            if prompt_flag_file.exists():
                user_prompt_file = self.prompts_dir / f"user_{user_id}.txt"
                user_prompt_file.write_text(user_msg, encoding="utf-8")
                prompt_flag_file.unlink()
                reply = "✅ 系統提示詞已更新！"
                self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                return

            # 功能 2️⃣：進入設定提示詞流程
            if user_msg == "設定提示詞":
                current_prompt = self.get_system_prompt(user_id)
                prompt_flag_file.write_text("awaiting", encoding="utf-8")
                reply = f"🔧 現在的提示詞如下：\n\n{current_prompt}\n\n請輸入你想要變更的新提示詞："
                self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                return

            # 功能 3️⃣：清除提示詞
            if user_msg == "清除提示詞":
                prompt_path = self.prompts_dir / f"user_{user_id}.txt"
                if prompt_path.exists():
                    prompt_path.unlink()
                    reply = "✅ 已清除使用者提示詞，恢復為預設提示詞。"
                else:
                    reply = "ℹ️ 你尚未自訂提示詞，已使用預設提示詞。"
                self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                return

            # 功能 4️⃣：清除聊天紀錄
            if user_msg == "/bye":
                history_path = self.history_dir / f"user_{user_id}.json"
                if history_path.exists():
                    history_path.unlink()
                reply = "🗑️ 已清除你的聊天紀錄，從頭開始囉！"
                self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                return

            # 🧠 一般對話流程
            # 使用者輸入的內容是純文字
            user_content = [user_msg]
            ai_reply = self.get_ai_response(user_id, user_content)
            self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ai_reply))

        # ⭐️ 8. 新增圖片訊息處理器
        @self.handler.add(MessageEvent, message=ImageMessage)
        def handle_image_message(event):
            user_id = event.source.user_id
            message_id = event.message.id
            logger.info(f"收到來自 {user_id} 的圖片訊息，ID: {message_id}")

            try:
                # 從 LINE 下載圖片
                message_content = self.line_bot_api.get_message_content(message_id)
                image_bytes = message_content.content
                
                # 使用 PIL 開啟圖片
                img = Image.open(io.BytesIO(image_bytes))
                
                # 為了歷史紀錄，將圖片存到本地
                image_path = self.image_dir / f"{user_id}_{message_id}.png"
                img.save(image_path, "PNG") # 存成 PNG 格式
                
                # 準備要傳給 Gemini 的內容，包含圖片和引導性文字
                # 這樣可以讓 AI 的回答更符合我們的預期
                user_content = [
                    "請詳細描述這張圖片的內容。如果圖片中有文字，也請一併列出。",
                    img
                ]

                ai_reply = self.get_ai_response(user_id, user_content)
                
                # 在更新歷史紀錄時，我們儲存的是圖片的路徑，而不是龐大的圖片本身
                # 注意：這裡的 user_content 傳給 get_ai_response 時是包含圖片物件的
                # 但儲存時，我們把圖片物件換成它的路徑字串
                history = self.load_chat_history(user_id)
                history.append({"role": "user", "parts": [user_content[0], str(image_path)]})
                history.append({"role": "model", "parts": [ai_reply]})
                history = self.manage_chat_history(history)
                self.save_chat_history(user_id, history)

                self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ai_reply))

            except Exception as e:
                logger.error(f"處理圖片訊息時發生錯誤: {e}")
                self.line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="抱歉，處理圖片時發生了一點問題。")
                )

    def run(self, host="0.0.0.0", port=5566):
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    bot = ChatBot()
    bot.run()