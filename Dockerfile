# 使用輕量的 Python 映像
FROM python:3.12.9-slim

# 設定工作目錄
WORKDIR /app

# 複製檔案
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 開放 Flask port
EXPOSE 5566

# 執行 app.py
CMD ["python", "app.py"]
