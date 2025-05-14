print("🔥 This is the updated FastAPI backend!")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(f"🛠 Running main.py from: {os.path.abspath(__file__)}")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chatbot_router  # after CORS import

app = FastAPI()

# ✅ Add CORS middleware BEFORE including any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "✅ This is your updated backend!"}
print("🔥 This is the updated FastAPI backend!")
# ✅ Add a simple root route to test
# ✅ Now include your router
app.include_router(chatbot_router)
