# DOCX RAG Chatbot (LangChain + Streamlit)

最小可用 RAG 聊天机器人，适配 Streamlit Community Cloud。

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cloud 部署
1. 推送到 GitHub。
2. 在 https://share.streamlit.io 新建 App，Main file 选 `app.py`。
3. 在 **Settings → Secrets** 写入：
```toml
OPENAI_BASE_URL = "https://www.dmxapi.cn/v1"
OPENAI_API_KEY  = "你的DMXkey"
```
4. 部署完成后获得可分享链接。

---

