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
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    StickerMessageContent,
    AudioMessageContent
)

from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import protos # <<< 確保 protos 被匯入
from PIL import Image
import io
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(self):
        self.app = Flask(__name__)
        self.load_environment()
        self.setup_line_bot()
        self.setup_gemini_config() # 僅設定 config，模型實例化移至 get_ai_response
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
        self.audio_dir = Path("audios")
        self.audio_dir.mkdir(exist_ok=True)
        
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

    def setup_gemini_config(self): # 改名以反映其作用
        genai.configure(api_key=self.gemini_api_key)
        self.generation_config_dict = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

    def get_system_prompt(self, user_id):
        user_prompt_file = self.prompts_dir / f"user_{user_id}.txt"
        if user_prompt_file.exists():
            return user_prompt_file.read_text(encoding="utf-8").strip()
        
        default_prompt_path = Path(self.system_prompt_file)
        if default_prompt_path.exists():
            return default_prompt_path.read_text(encoding="utf-8").strip()
        
        return "你是一個友善、溫暖且樂於助人的AI助手。請使用繁體中文與使用者互動，保持簡潔、親切、同理心的語調。如果收到圖片、貼圖或語音，請描述它們或理解其內容，並根據上下文回應。"

    def load_chat_history(self, user_id):
        path = self.history_dir / f"user_{user_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"讀取歷史檔案失敗或格式錯誤: {path}")
                return []
        return []

    def save_chat_history(self, user_id, history):
        path = self.history_dir / f"user_{user_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def manage_chat_history(self, history):
        total_tokens = sum(len(str(part)) for turn in history for part in turn.get('parts', [])) 
        
        while total_tokens > self.max_history_tokens and len(history) > 1:
            removed_turn = history.pop(0)
            removed_tokens_this_turn = 0
            for part in removed_turn.get('parts', []):
                removed_tokens_this_turn += len(str(part))
                if isinstance(part, str):
                    part_path = Path(part)
                    if part.startswith(str(self.image_dir)) or part.startswith(str(self.audio_dir)):
                        try:
                            part_path.unlink(missing_ok=True)
                            logger.info(f"已從歷史記錄管理器中刪除媒體檔案: {part}")
                        except OSError as e:
                            logger.warning(f"刪除歷史媒體檔案失敗 {part}: {e}")
            total_tokens -= removed_tokens_this_turn
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
                
                parts_for_gemini = []
                for part_data in turn.get('parts', []):
                    if isinstance(part_data, str) and part_data.startswith(str(self.image_dir)):
                        try:
                            img = Image.open(part_data)
                            parts_for_gemini.append(img)
                        except FileNotFoundError:
                            logger.warning(f"歷史圖片未找到: {part_data}, 以文字替代")
                            parts_for_gemini.append(f"(圖片已遺失: {Path(part_data).name})")
                        except Exception as e:
                            logger.error(f"載入歷史圖片失敗 {part_data}: {e}", exc_info=True)
                            parts_for_gemini.append(f"(載入歷史圖片錯誤: {Path(part_data).name})")
                    elif isinstance(part_data, str) and part_data.startswith(str(self.audio_dir)):
                        # 音訊路徑在歷史中僅用於標識和清理，不作為原始音訊傳遞給 history
                        # AI對音訊的理解已轉為文字存在於assistant的回應中
                        # 如果需要，這裡可以添加如 "(用戶曾發送過音訊)" 的標記文字
                        # parts_for_gemini.append(f"(歷史音訊檔案: {Path(part_data).name})")
                        pass # 通常不將歷史音訊檔案路徑直接加入gemini的歷史內容
                    else:
                        parts_for_gemini.append(part_data)
                
                if parts_for_gemini and role in ["user", "model"]:
                    gemini_history.append({'role': role, 'parts': parts_for_gemini})
            
            if not user_content:
                logger.error("get_ai_response收到的user_content為空")
                return "抱歉，無法處理空的請求。"

            logger.debug(f"向 Gemini 發送歷史: {len(gemini_history)} turns. 用戶內容類型: {[type(p) for p in user_content]}")
            
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_content) # user_content 可以包含 protos.Part
            
            return response.text.strip() if response.text else "抱歉，我暫時無法回應。"
        except Exception as e:
            logger.error(f"Gemini API 回應失敗：{e}", exc_info=True)
            error_message = str(e)
            if hasattr(e, 'message'):
                error_message = e.message
            
            if "API key not valid" in error_message:
                return "Gemini API 金鑰設定錯誤，請檢查設定。"
            if "quota" in error_message.lower():
                return "已達到 Gemini API 的使用額度限制，請稍後再試。"
            if "SAFETY" in error_message.upper():
                 logger.warning(f"Gemini 安全性封鎖: {error_message}")
                 return "抱歉，您的請求可能包含不適當的內容，我無法處理。"
            return "發生錯誤，請稍後再試～"

    def setup_routes(self):
        
        def background_task(user_id, event_type, data):
            try:
                user_content_for_gemini = []
                storable_parts_for_history = []

                if event_type == 'text':
                    user_text = data
                    user_content_for_gemini = [user_text]
                    storable_parts_for_history = [user_text]

                elif event_type == 'image':
                    image_bytes = data 
                    image_obj = Image.open(io.BytesIO(image_bytes))
                    filename_ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    filename = f"userimg_{user_id}_{filename_ts}.png"
                    image_path = self.image_dir / filename
                    image_obj.save(image_path, 'PNG')
                    
                    prompt_text = "請描述這張用戶上傳的圖片。如果圖片中有文字，也請一併列出。"
                    user_content_for_gemini = [prompt_text, image_obj]
                    storable_parts_for_history = [prompt_text, str(image_path)]

                elif event_type == 'sticker':
                    sticker_image_bytes, package_id, sticker_id = data
                    sticker_prompt_text = f"用戶發送了一個 LINE 貼圖 (Package ID: {package_id}, Sticker ID: {sticker_id})。"

                    if sticker_image_bytes:
                        try:
                            sticker_obj = Image.open(io.BytesIO(sticker_image_bytes))
                            filename_ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                            filename = f"sticker_{user_id}_{package_id}_{sticker_id}_{filename_ts}.png"
                            sticker_path = self.image_dir / filename
                            sticker_obj.save(sticker_path, 'PNG')
                            
                            sticker_prompt_text_full = sticker_prompt_text + "圖片內容如下。請描述這個貼圖並推測用戶的情感或意圖。"
                            user_content_for_gemini = [sticker_prompt_text_full, sticker_obj]
                            storable_parts_for_history = [sticker_prompt_text_full, str(sticker_path)]
                            logger.info(f"貼圖圖片已處理並保存於: {sticker_path}")
                        except Exception as e:
                            logger.error(f"處理下載的貼圖圖片失敗: {e}", exc_info=True)
                            sticker_prompt_text_full = sticker_prompt_text + "但無法顯示圖片。請根據 ID 推測含義。"
                            user_content_for_gemini = [sticker_prompt_text_full]
                            storable_parts_for_history = [sticker_prompt_text_full]
                    else:
                        sticker_prompt_text_full = sticker_prompt_text + "但無法獲取其實際圖片。請根據 ID 推測其可能的含義和用戶情感。"
                        user_content_for_gemini = [sticker_prompt_text_full]
                        storable_parts_for_history = [sticker_prompt_text_full]
                
                elif event_type == 'audio':
                    audio_bytes, message_id_for_filename = data
                    mime_type = "audio/m4a"
                    
                    filename = f"useraudio_{user_id}_{message_id_for_filename}.m4a"
                    audio_path = self.audio_dir / filename
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    logger.info(f"音訊檔案已儲存於: {audio_path}")

                    audio_prompt_text = "用戶發送了一段語音，請理解其內容並作出回應。例如，如果是問題請回答，如果是陳述請給予回應。"
                    
                    # --- 使用 protos.Blob 和 protos.Part 處理音訊 ---
                    try:
                        audio_blob = protos.Blob(mime_type=mime_type, data=audio_bytes)
                        audio_part_for_gemini = protos.Part(inline_data=audio_blob)
                        logger.info("成功使用 protos.Blob 和 protos.Part 構造音訊部分。")
                    except Exception as e:
                        logger.error(f"直接構造 protos.Blob/Part 失敗: {e}", exc_info=True)
                        self.messaging_api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[TextMessage(text="抱歉，處理您的語音訊息時內部發生錯誤。")]
                            )
                        )
                        return # 終止此任務
                    # --- 修改結束 ---
                    
                    user_content_for_gemini = [audio_prompt_text, audio_part_for_gemini]
                    storable_parts_for_history = [
                        f"用戶發送了一段語音（檔案：{filename}）。", 
                        str(audio_path)
                    ]

                else:
                    logger.warning(f"未知的事件類型給 background_task: {event_type}")
                    self.messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text="抱歉，收到了未知的訊息類型。")]
                        )
                    )
                    return
                
                if not user_content_for_gemini: # 再次檢查，以防 audio 處理提前返回
                    logger.warning(f"事件類型 {event_type} 未能成功準備 user_content_for_gemini (可能在 audio 處理中失敗)")
                    # 可以不推播，因為 audio 處理失敗時已推播
                    return


                history = self.load_chat_history(user_id)
                ai_reply = self.get_ai_response(user_id, history, user_content_for_gemini)

                history.append({"role": "user", "parts": storable_parts_for_history})
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
                logger.error(f"背景任務失敗 ({event_type}): {e}", exc_info=True)
                try:
                    self.messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text="抱歉，處理您的請求時發生了一點問題。")]
                        )
                    )
                except Exception as push_e:
                    logger.error(f"背景任務失敗後，推播錯誤訊息也失敗: {push_e}", exc_info=True)


        @self.app.route("/callback", methods=["POST"])
        def callback():
            signature = request.headers["X-Line-Signature"]
            body = request.get_data(as_text=True)
            try:
                self.handler.handle(body, signature)
            except InvalidSignatureError:
                logger.warning("無效的簽名")
                abort(400)
            except Exception as e:
                logger.error(f"Callback 處理時發生錯誤: {e}", exc_info=True)
                abort(500)
            return "OK"

        @self.handler.add(MessageEvent, message=TextMessageContent)
        def handle_text_message(event):
            user_id = event.source.user_id
            user_msg = event.message.text.strip()
            
            def reply_sync(text):
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
                    prompt_path.unlink(missing_ok=True)
                    reply_sync("✅ 已清除使用者提示詞，恢復為預設提示詞。")
                else:
                    reply_sync("ℹ️ 你尚未自訂提示詞，已使用預設提示詞。")
                return

            if user_msg == "/bye":
                history_path = self.history_dir / f"user_{user_id}.json"
                if history_path.exists():
                    try:
                        old_history_content = self.load_chat_history(user_id)
                        for turn in old_history_content:
                            for part_data in turn.get('parts', []):
                                if isinstance(part_data, str):
                                    part_path_obj = Path(part_data)
                                    if part_data.startswith(str(self.image_dir)) or \
                                       part_data.startswith(str(self.audio_dir)):
                                        try:
                                            part_path_obj.unlink(missing_ok=True)
                                            logger.info(f"清除歷史時刪除媒體檔案: {part_data}")
                                        except OSError as e:
                                            logger.warning(f"清除歷史媒體檔案時發生錯誤 {part_data}: {e}")
                        history_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.error(f"清除歷史檔案 {history_path} 失敗: {e}")
                
                reply_sync("🗑️ 已清除你的聊天紀錄與相關媒體檔案，從頭開始囉！")
                return

            logger.info(f"收到來自 {user_id} 的文字訊息，將啟動背景AI處理：{user_msg}")
            threading.Thread(target=background_task, args=(user_id, 'text', user_msg)).start()

        @self.handler.add(MessageEvent, message=ImageMessageContent)
        def handle_image_message(event):
            user_id = event.source.user_id
            message_id = event.message.id
            logger.info(f"收到來自 {user_id} 的圖片訊息 (ID: {message_id})，準備下載並啟動背景AI處理")
            try:
                message_content_bytes = self.messaging_api_blob.get_message_content(message_id=message_id)
                threading.Thread(target=background_task, args=(user_id, 'image', message_content_bytes)).start()
            except Exception as e:
                logger.error(f"處理圖片訊息時下載失敗: {e}", exc_info=True)
                try:
                    self.messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text="抱歉，下載您傳送的圖片時發生問題，請稍後再試。")]
                        )
                    )
                except Exception as push_e:
                    logger.error(f"推播圖片下載失敗訊息時也失敗: {push_e}")

        @self.handler.add(MessageEvent, message=StickerMessageContent)
        def handle_sticker_message(event):
            user_id = event.source.user_id
            package_id = event.message.package_id
            sticker_id = event.message.sticker_id
            
            logger.info(f"收到來自 {user_id} 的貼圖訊息：Package ID: {package_id}, Sticker ID: {sticker_id}")
            
            sticker_image_bytes = None
            urls_to_try = [
                f"https://stickershop.line-scdn.net/stickershop/v1/sticker/{sticker_id}/ANDROID/sticker.png",
                f"https://stickershop.line-scdn.net/stickershop/v1/sticker/{sticker_id}/IOS/sticker.png",
                f"https://stickershop.line-scdn.net/stickershop/v1/sticker/{sticker_id}/STATIC/sticker.png"
            ]

            for s_url in urls_to_try:
                try:
                    response = requests.get(s_url, timeout=10)
                    if response.status_code == 200 and response.content:
                        sticker_image_bytes = response.content
                        logger.info(f"成功從 {s_url} 下載貼圖圖片 ({len(sticker_image_bytes)} bytes)")
                        break 
                    else:
                        logger.warning(f"下載貼圖圖片失敗 ({s_url}): HTTP {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"下載貼圖圖片時發生網路請求錯誤 ({s_url}): {e}")
            
            if not sticker_image_bytes:
                logger.warning(f"無法下載貼圖 {package_id}/{sticker_id} 的圖片，將僅使用文字描述。")

            threading.Thread(target=background_task, args=(user_id, 'sticker', (sticker_image_bytes, package_id, sticker_id))).start()
        
        @self.handler.add(MessageEvent, message=AudioMessageContent)
        def handle_audio_message(event):
            user_id = event.source.user_id
            message_id = event.message.id
            duration_ms = event.message.duration

            logger.info(f"收到來自 {user_id} 的語音訊息 (ID: {message_id}, 時長: {duration_ms}ms)，準備下載並啟動背景AI處理")
            
            try:
                message_content_bytes = self.messaging_api_blob.get_message_content(message_id=message_id)
                threading.Thread(target=background_task, args=(user_id, 'audio', (message_content_bytes, message_id))).start()
            except Exception as e:
                logger.error(f"處理語音訊息時下載失敗: {e}", exc_info=True)
                try:
                    self.messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text="抱歉，下載您傳送的語音時發生問題，請稍後再試。")]
                        )
                    )
                except Exception as push_e:
                    logger.error(f"推播語音下載失敗訊息時也失敗: {push_e}")


    def run(self, host="0.0.0.0", port=5566):
        logger.info(f"聊天機器人啟動於 http://{host}:{port}")
        self.app.run(host=host, port=port, threaded=True) 

if __name__ == "__main__":
    bot = ChatBot()
    port = int(os.environ.get("PORT", 5566))
    bot.run(host="0.0.0.0", port=port)