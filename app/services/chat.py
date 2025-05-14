# backend/app/services/chat.py
from app.services.rag import get_rag_response
from app.utils.redis_handler import get_redis_client

redis = get_redis_client()

async def get_answer_with_context(session_id: str, user_message: str):
    response = get_rag_response(user_message)
    
    redis.rpush(session_id, f"User: {user_message}")
    redis.rpush(session_id, f"Bot: {response}")
    
    return response

def get_session_history(session_id: str):
    return [msg.decode() for msg in redis.lrange(session_id, 0, -1)]

def clear_session_history(session_id: str):
    redis.delete(session_id)
