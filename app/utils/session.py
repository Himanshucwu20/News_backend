# app/utils/session.py
import redis
import uuid
import json

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Create a new session
def create_session():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    redis_client.set(session_id, json.dumps({"history": []}))  # Initialize an empty history
    return session_id

# Get the history for a given session
def get_session_history(session_id: str):
    history = redis_client.get(session_id)
    if not history:
        return None
    return json.loads(history)["history"]

# Clear a session by deleting it from Redis
def clear_session(session_id: str):
    if redis_client.exists(session_id):
        redis_client.delete(session_id)
        return True
    return False
