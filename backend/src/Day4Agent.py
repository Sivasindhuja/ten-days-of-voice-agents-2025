import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    MetricsCollectedEvent,
    metrics,
)

from livekit.plugins import silero, google, murf, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tools.tutor_tools import (
    load_tutor_content,
    get_summary,
    get_sample_question,
)

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ------------------- 1️⃣ Define Agent Behavior -------------------

class TutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are an Active Recall Tutor. You help the student learn by using three modes:
1. learn – explain a concept using summary from our content file.
2. quiz – ask questions about the concept.
3. teach_back – ask the student to explain the concept in their own words.

Use this small content file already loaded:
Each concept has: id, title, summary, sample_question.

Example:
If user says "I want to learn loops", you do:
Mode: learn
Concept: loops
→ Explain using the summary.

Use these voices:
🎓 Learn mode → Murf voice "Matthew"
📝 Quiz mode → Murf voice "Alicia"
🗣 Teach-back mode → Murf voice "Ken"

Important Rules:
- ALWAYS respect the current mode.
- Let the user switch modes naturally by asking things like:
  "Now quiz me", "Let me explain", "teach_back", etc.
- In teach_back mode, give simple feedback like:
  "Great explanation!" or "You missed an important part..."

Keep responses short, friendly, and engaging.
"""
        )

# --------------------- 2️⃣ Prewarm: Load VAD ----------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# --------------------- 3️⃣ Entrypoint ----------------------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    content = load_tutor_content()
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # default voice
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        tools=[],  # No function calls needed yet
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage_summary():
        logger.info(f"Usage Summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage_summary)

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# --------------------- 4️⃣ Main Runner ----------------------

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
