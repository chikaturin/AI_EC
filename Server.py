from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import uvicorn
import json

load_dotenv()
AI_KEY = os.getenv("AI_API_KEY")
if not AI_KEY:
    raise ValueError("AI_API_KEY not found in environment variables")

genai.configure(api_key=AI_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

def chatwithAIGemini(question: str):
    relevant_docs = db.similarity_search(question)
    if not relevant_docs:
        return {"message": "Không tìm thấy thông tin phù hợp."}

    context_data = []
    for d in relevant_docs:
        metadata = d.metadata if isinstance(d.metadata, dict) else {}
        id_value = metadata.get("id")
        original_content = metadata.get("original_content", {})
        text_summary = original_content.get("text_summary")

        if id_value and text_summary:
            context_data.append({"ID": id_value, "Text": text_summary})
    
    if not context_data:
        return {"message": "Không có dữ liệu phù hợp để trả lời."}

    prompt_gemini = f"""
    Bạn là một chuyên gia về linh kiện và cấu hình máy tính, có kiến thức chi tiết về khả năng tương thích và tối ưu hóa.
    Dưới đây là thông tin sản phẩm từ ngữ cảnh:

    {context_data}

    **Câu hỏi**: {question}

    Hãy chỉ tập trung vào việc giới thiệu sản phẩm hay ho nhất từ dữ liệu, không quan tâm đến thông tin không liên quan.
    Trả lời dưới dạng JSON như sau:

    ```json
    {{
    "ID": "ID của sản phẩm",
    "Note": "Lời khuyên và đánh giá chi tiết về sản phẩm",
    "Image": "Hình ảnh sản phẩm"
    }}
    ```
    """
    
    response = model_gemini.generate_content(prompt_gemini)
    return response.text

@app.post("/chat")
def chat(request: QueryRequest):
    try:
        response = chatwithAIGemini(request.question)
        cleanedResult = response.replace('```json', '').replace('```', '').strip()
        json_result = json.loads(cleanedResult)
        return {json_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Chat Server!"}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
