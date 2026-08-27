# test_sentiment_synthesis.py
import asyncio, uuid
from dotenv import load_dotenv
load_dotenv()
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.sentiment_synthesis import sentiment_synthesis_agent, build_sentiment_prompt

sample_youtube = {
    "abc123": {
        "title": "Dune Part Three - Official Trailer",
        "view_count": 1_000_000, "like_count": 50_000, "comment_count": 2_000,
        "comments": [
            {"text": "This looks incredible, can't wait!", "like_count": 500},
            {"text": "Pacing looked rushed in this trailer", "like_count": 120},
        ],
    }
}

async def main():
    prompt = build_sentiment_prompt(
        "Critics praised the visuals but noted pacing concerns.", sample_youtube
    )
    session_service = InMemorySessionService()
    user_id, session_id = "test_user", f"test_{uuid.uuid4().hex[:6]}"
    await session_service.create_session(app_name="filmecho", user_id=user_id, session_id=session_id)
    runner = Runner(agent=sentiment_synthesis_agent, app_name="filmecho", session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)

asyncio.run(main())