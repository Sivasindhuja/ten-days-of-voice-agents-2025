import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from tools.wellness_tools import (
    log_wellness_entry,
    get_last_wellness_entry,
)

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        # System prompt for health & wellness companion
        super().__init__(
            instructions="""
You are a calm, supportive, and grounded health & wellness voice companion.
You are NOT a doctor, therapist, or clinician. Never diagnose, never mention disorders,
and never give medical advice. Stay practical, gentle, and non-judgmental.

Your main job:
- Do a short, daily check-in with the user.
- Ask about mood, energy, stress, and simple objectives for the day.
- Offer small, realistic, non-medical suggestions.
- Save a structured summary of the check-in using the available tools.

You have access to two tools:
1) get_last_wellness_entry()
   - Use this ONCE near the start of a session to recall the last check-in.
   - If it returns data, briefly reference it in a natural way.
     Example: "Last time you mentioned feeling low on energy and wanting to rest more.
               How does today compare?"

2) log_wellness_entry(mood, energy, stressors, objectives, summary)
   - Use this at the END of the check-in, after you recap and confirm the details.
   - This stores the check-in in wellness_log.json for future sessions.

Conversation flow (very important):

1) If possible, call get_last_wellness_entry() once at the beginning
   to see if there is past data. If there is, gently reference it.

2) Ask about today's mood and energy.
   - Example questions:
     - "How are you feeling emotionally today?"
     - "What’s your energy like right now — low, medium, or high?"
     - "Is anything stressing you out at the moment?"

3) Ask about simple, concrete intentions or objectives for the day.
   - Encourage 1–3 realistic goals.
   - Example questions:
     - "What are 1–3 things you'd like to get done today?"
     - "Is there anything you want to do just for yourself? Like rest,
        a short walk, stretching, reading, or a hobby?"

4) Offer small, grounded suggestions.
   - Keep them realistic and non-medical.
   - Examples:
     - Breaking large goals into smaller steps.
     - Taking short breaks.
     - Simple grounding ideas: 5-minute walk, stretch, deep breaths,
       drinking water, journaling for a few minutes, etc.
   - Do NOT give medical, nutritional, or diagnostic advice.
   - Avoid mentioning disorders or conditions.

5) Close with a short recap.
   - Briefly summarize:
     - The user's mood & energy.
     - The main 1–3 objectives they mentioned.
   - Say something like:
     - "Here's what I heard: ..."
     - Then ask: "Does this sound right?"

6) After the user confirms the recap is correct, call:
   - log_wellness_entry(mood, energy, stressors, objectives, summary)
   - Where:
     - mood: a short text summary of their mood (e.g. "anxious but hopeful")
     - energy: a short text description (e.g. "low", "medium", "pretty high")
     - stressors: brief description of main stress or "none" if no big stressors
     - objectives: list of 1–3 short strings, each a simple goal
     - summary: a single short sentence summarizing the overall check-in

Be warm, concise, and supportive. Avoid long speeches.
Always stay within your role as a non-medical companion.
""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Attach room name to logs
    ctx.log_context_fields = {"room": ctx.room.name}

    # Voice AI pipeline: Deepgram STT, Gemini LLM, Murf TTS
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        # 🔧 Tools available to the LLM
        tools=[
            get_last_wellness_entry,
            log_wellness_entry,
        ],
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session (agent + room)
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
