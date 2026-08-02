# assist_server.py
"""
Guiden Real-Time Navigation Server
────────────────────────────────────
- Persistent WebSocket per user session
- Continuous frame analysis (proactive) with frame diff filtering
- Reactive Q&A on question events
- Goal-directed navigation ("take me to white chair")
- In-memory context with optional mem0 persistence
- ElevenLabs TTS
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import httpx
import numpy as np
import replicate
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel


load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
MEM0_API_KEY        = os.getenv("MEM0_API_KEY", "")

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guiden")

app = FastAPI(title="Guiden")

# ─── Try to import mem0 (optional) ───────────────────────────────────────────
try:
    from mem0 import MemoryClient
    _mem0_client = MemoryClient(api_key=MEM0_API_KEY) if MEM0_API_KEY else None
    logger.info("mem0 client initialized" if _mem0_client else "mem0 skipped (no key)")
except ImportError:
    _mem0_client = None
    logger.info("mem0 not installed, using in-memory context only")


# ─── Frame Diff Filter ────────────────────────────────────────────────────────

class FrameFilter:
    """
    Prevents sending redundant frames to the AI.
    Only forwards a frame when:
      1. Minimum time interval has elapsed AND
      2. The scene has changed enough (pixel diff threshold)
    
    This can cut API calls by 80-95% during static/slow scenes.
    """

    def __init__(
        self,
        change_threshold: float = 0.15,   # 15% pixel change required
        min_send_interval: float = 2.5,    # never faster than 2.5s
        max_send_interval: float = 8.0,    # force send after 8s of silence
        diff_resize: tuple = (160, 120),   # tiny resolution for fast compare
        diff_pixel_cutoff: int = 25,       # pixel diff value to count as "changed"
    ):
        self.change_threshold   = change_threshold
        self.min_send_interval  = min_send_interval
        self.max_send_interval  = max_send_interval
        self.diff_resize        = diff_resize
        self.diff_pixel_cutoff  = diff_pixel_cutoff

        self._last_sent_frame: Optional[np.ndarray] = None
        self._last_sent_time: float = 0.0
        self._frames_dropped: int = 0      # telemetry
        self._frames_sent: int = 0         # telemetry

    # ── Public ────────────────────────────────────────────────────────────────

    def should_send(self, jpeg_b64: str) -> tuple[bool, float]:
        """
        Returns (should_send, change_ratio).
        change_ratio=1.0 means forced send (timeout), 0.0 means no change.
        """
        now = time.time()
        elapsed = now - self._last_sent_time

        # Hard minimum: never faster than min_send_interval
        if elapsed < self.min_send_interval:
            self._frames_dropped += 1
            return False, 0.0

        # Decode + resize frame for comparison
        frame_small = self._decode_small(jpeg_b64)
        if frame_small is None:
            # Decode failed — send anyway to not block navigation
            self._record_sent(now, None)
            return True, 1.0

        # Force-send if we haven't sent in max_send_interval
        if elapsed >= self.max_send_interval:
            self._record_sent(now, frame_small)
            return True, 1.0

        # First frame ever
        if self._last_sent_frame is None:
            self._record_sent(now, frame_small)
            return True, 1.0

        # Compute pixel-level change ratio
        diff = cv2.absdiff(frame_small, self._last_sent_frame)
        changed_pixels = np.count_nonzero(diff > self.diff_pixel_cutoff)
        change_ratio = changed_pixels / diff.size

        if change_ratio >= self.change_threshold:
            self._record_sent(now, frame_small)
            return True, change_ratio

        self._frames_dropped += 1
        return False, change_ratio

    @property
    def stats(self) -> dict:
        total = self._frames_sent + self._frames_dropped
        drop_pct = (self._frames_dropped / total * 100) if total else 0
        return {
            "sent": self._frames_sent,
            "dropped": self._frames_dropped,
            "drop_rate_pct": round(drop_pct, 1),
        }

    def reset(self):
        """Call when user resumes after stop/pause to re-baseline."""
        self._last_sent_frame = None
        self._last_sent_time = 0.0

    # ── Private ───────────────────────────────────────────────────────────────

    def _decode_small(self, jpeg_b64: str) -> Optional[np.ndarray]:
        try:
            img_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                return None
            return cv2.resize(frame, self.diff_resize, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.debug(f"FrameFilter decode error: {e}")
            return None

    def _record_sent(self, t: float, frame: Optional[np.ndarray]):
        self._last_sent_time = t
        self._last_sent_frame = frame
        self._frames_sent += 1


# ─── Data Models ─────────────────────────────────────────────────────────────

class NavMode(str, Enum):
    IDLE          = "idle"
    FREE_ROAM     = "free_roam"
    GOAL_DIRECTED = "goal_directed"


@dataclass
class TaskPlan:
    """A structured navigation task broken into trackable steps."""
    raw_request: str          # "take me to the whiteboard"
    goal_object: str          # "whiteboard"
    goal_description: str     # fuller description for vision matching
    steps: list               # ["Turn left", "Walk 5 steps", "Stop at board"]
    current_step_idx: int = 0
    landmarks_to_find: list = field(default_factory=list)  # intermediate objects
    completed: bool = False
    started_at: float = field(default_factory=time.time)
    
    @property
    def current_step(self) -> str:
        if self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return ""
    
    def advance(self):
        self.current_step_idx += 1
        if self.current_step_idx >= len(self.steps):
            self.completed = True
    
    @property
    def progress_pct(self) -> int:
        if not self.steps:
            return 0
        return int((self.current_step_idx / len(self.steps)) * 100)




@dataclass
class SceneMemory:
    """Rolling window of what the AI has seen."""
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    obstacles_left: str = ""
    obstacles_right: str = ""
    obstacles_ahead: str = ""
    floor_clear: bool = True


@dataclass  
class NavigationContext:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = "default_user"

    mode: NavMode = NavMode.FREE_ROAM
    goal: Optional[str] = None
    goal_reached: bool = False
    
    # ── NEW: Full task plan ──────────────────────────────────
    active_task: Optional[TaskPlan] = None

    scene_history: deque = field(default_factory=lambda: deque(maxlen=5))
    last_scene: Optional[SceneMemory] = None
    conversation: list = field(default_factory=list)

    last_proactive_time: float = 0.0
    proactive_interval: float = 2.5

    is_stopped: bool = False
    last_spoken: str = ""
    consecutive_silent: int = 0

    def should_proact(self) -> bool:
        if self.is_stopped:
            return False
        return time.time() - self.last_proactive_time >= self.proactive_interval

    def mark_proact(self):
        self.last_proactive_time = time.time()

    def adapt_interval(self, urgency: str, same_as_last: bool):
        if urgency == "critical":
            self.proactive_interval = 1.0   # faster during tasks
        elif urgency == "high":
            self.proactive_interval = 1.5
        elif urgency == "task_step":         # new urgency for step transitions
            self.proactive_interval = 1.2
        elif same_as_last:
            self.proactive_interval = min(self.proactive_interval * 1.3, 6.0)
        else:
            self.proactive_interval = {
                "medium": 2.0,
                "low":    3.0,
            }.get(urgency, 2.5)

    def set_goal(self, goal: str):
        self.goal = goal
        self.mode = NavMode.GOAL_DIRECTED
        self.goal_reached = False
        self.is_stopped = False

    def set_task(self, task: TaskPlan):
        """Full task with plan — richer than simple goal."""
        self.active_task = task
        self.goal = task.goal_object
        self.mode = NavMode.GOAL_DIRECTED
        self.goal_reached = False
        self.is_stopped = False

    def clear_goal(self):
        self.goal = None
        self.active_task = None
        self.mode = NavMode.FREE_ROAM

    def stop(self):
        self.is_stopped = True

    def resume(self):
        self.is_stopped = False

    def add_exchange(self, role: str, content: str):
        self.conversation.append({
            "role": role,
            "content": content,
            "ts": time.time()
        })
        if len(self.conversation) > 24:
            self.conversation = self.conversation[-24:]

    @property
    def recent_scene_text(self) -> str:
        return self.last_scene.description if self.last_scene else "No scene analyzed yet."

    @property
    def history_summary(self) -> str:
        recent = self.conversation[-6:]
        lines = []
        for ex in recent:
            role = "User" if ex["role"] == "user" else "Guiden"
            lines.append(f"{role}: {ex['content']}")
        return "\n".join(lines) if lines else "None"
    
    @property
    def task_context(self) -> str:
        """Rich task context for prompt injection."""
        if not self.active_task:
            return ""
        t = self.active_task
        steps_display = "\n".join([
            f"  {'✓' if i < t.current_step_idx else ('→' if i == t.current_step_idx else '○')} Step {i+1}: {s}"
            for i, s in enumerate(t.steps)
        ])
        return f"""
ACTIVE TASK: "{t.raw_request}"
Target: {t.goal_object}
Progress: {t.progress_pct}% ({t.current_step_idx}/{len(t.steps)} steps)
Current step: {t.current_step}
Steps:
{steps_display}
Landmarks to find: {', '.join(t.landmarks_to_find) if t.landmarks_to_find else 'none specified'}"""


_TASK_PLAN_PROMPT = """The user said: "{request}"

Look at this camera image carefully. Create a navigation task plan.

Return ONLY valid JSON:
{{
  "goal_object": "the specific thing they want to reach (concise noun)",
  "goal_description": "detailed visual description to identify it in frames",
  "is_visible_now": true or false,
  "initial_direction": "where to look/go first based on what you see",
  "landmarks_to_find": ["intermediate objects that mark the path"],
  "steps": [
    "Step 1 instruction",
    "Step 2 instruction",
    "..."
  ],
  "first_spoken": "warm, confident first instruction to say out loud (2 sentences max)"
}}

STEP WRITING RULES:
- Each step = one clear physical action
- Use landmark-based guidance: "Walk toward the door on your left"
- Include distance estimates: "about 4 steps forward"
- Max 5 steps (merge small moves)
- Last step always = confirmation of arrival

If goal is NOT visible:
- first step = "Turn slowly left/right to scan for [goal]"
- Keep steps general until goal found"""


def _task_plan_prompt(request: str) -> str:
    return _TASK_PLAN_PROMPT.format(request=request)


# ─── Task Navigation Prompt (replaces proactive during task) ─────────────────

_TASK_NAV_PROMPT = """You are guiding a blind user through a task step-by-step.

TASK: "{goal_object}"
CURRENT STEP ({step_num}/{total_steps}): "{current_step}"
STEP DESCRIPTION: {goal_description}
LANDMARKS TO FIND: {landmarks}

Analyze this frame and return ONLY valid JSON:
{{
  "should_speak": true or false,
  "urgency": "critical" | "high" | "task_step" | "medium" | "low" | "silent",
  "step_completed": true or false,
  "goal_reached": true or false,
  "spoken_text": "what to say (empty if should_speak=false)",
  "confidence": 0.0 to 1.0,
  "scene": {{
    "ahead": "what is directly ahead",
    "left": "left side",
    "right": "right side", 
    "floor_clear": true or false,
    "summary": "one-line description",
    "target_visible": true or false,
    "target_location": "where is the target if visible"
  }}
}}

STEP COMPLETION RULES:
- step_completed=true ONLY when user has physically completed this step's action
- goal_reached=true ONLY when target fills bottom-third of frame OR user is right at it
- If step involves walking: confirm user has moved (scene changed significantly)
- If step involves turning: confirm new direction is visible

URGENCY RULES:
- critical: obstacle in path OR user about to go wrong way
- task_step: user completed a step, next step starting
- high: target now visible for first time
- medium: progress update, on track
- low: confirming steady progress
- silent: no change, save bandwidth

SPEAK RULES:
- If step_completed: announce next step clearly
- If goal_reached: "Stop. You have reached [goal]. It is right in front of you."
- If obstacle: warn FIRST, then redirect to task
- Keep it under 2 sentences unless critical"""


PRODUCT_ANALYSIS_SYSTEM = """You are a helpful product assistant for blind and visually impaired users.
You analyze product images and answer questions about them clearly and concisely.

RULES:
- Speak naturally as if talking to someone. No markdown, no bullet points, no lists.
- Keep responses under 4 sentences unless the user asks for detailed information.
- If you can read text/labels on the product, read them accurately.
- For food products: mention key ingredients, allergens, nutritional highlights.
- For medications: read the name, dosage, and any warnings visible.
- For household items: describe what it is and how to use it.
- Be warm, helpful, and direct.
- If you cannot identify the product clearly, say so honestly.
- ALWAYS address the user's specific question first, then add helpful context.
- For allergy questions: be VERY careful and thorough. Check ALL visible ingredients.
- If you cannot read all ingredients clearly, WARN the user about uncertainty."""

PRODUCT_IDENTIFY_PROMPT = """Look at this product image carefully.

Identify:
1. What product this is (brand, name, type)
2. Any visible text, labels, ingredients, warnings
3. Key details a blind person would need to know

Provide a brief, spoken-style description in 2-3 sentences."""

def build_question_prompt(question: str, product_context: str = "") -> str:
    ctx = ""
    if product_context:
        ctx = f"\nPrevious product identification: {product_context}\n"
    return f"""{ctx}
The user is holding a product in front of the camera and asked:
"{question}"

Look at the product image carefully and answer their specific question.
Be thorough but concise. Speak naturally in 2-4 sentences.
If they ask about allergies or ingredients, be VERY careful and check everything visible."""


# ─── Request/Response Models ─────────────────────────────────────────────────

class ProductQueryRequest(BaseModel):
    image_base64: str
    question: str
    session_id: Optional[str] = None


class ProductQueryResponse(BaseModel):
    text: str
    audio_base64: Optional[str] = None
    product_name: Optional[str] = None
    confidence: float = 0.0
    processing_time_ms: int = 0


class ProductIdentifyResponse(BaseModel):
    text: str
    audio_base64: Optional[str] = None
    product_name: str = "Unknown product"
    processing_time_ms: int = 0


_product_sessions: dict[str, dict] = {}


def _clean_expired_product_sessions(max_age_seconds: float = 900.0):
    """Purge product sessions older than 15 minutes to prevent RAM memory leaks."""
    now = time.time()
    expired = [
        sid for sid, data in _product_sessions.items()
        if now - data.get("timestamp", 0.0) > max_age_seconds
    ]
    for sid in expired:
        _product_sessions.pop(sid, None)



def _build_task_nav_prompt(ctx: NavigationContext) -> str:
    t = ctx.active_task
    if not t:
        return _PROACTIVE_PROMPT
    return _TASK_NAV_PROMPT.format(
        goal_object=t.goal_object,
        step_num=t.current_step_idx + 1,
        total_steps=len(t.steps),
        current_step=t.current_step,
        goal_description=t.goal_description,
        landmarks=", ".join(t.landmarks_to_find) if t.landmarks_to_find else "none",
    )

# ─── Prompts ─────────────────────────────────────────────────────────────────

def _build_system_prompt(ctx: NavigationContext) -> str:
    # Task context (richer than simple goal)
    task_section = ""
    if ctx.active_task and not ctx.active_task.completed:
        task_section = f"""
{ctx.task_context}
- Guide through ONE STEP AT A TIME
- Confirm step completion before moving to next
- Always re-orient to task after obstacle warnings
- If user seems confused: repeat current step differently
"""
    elif ctx.mode == NavMode.GOAL_DIRECTED and ctx.goal:
        task_section = f"""
═══ ACTIVE GOAL ═══
Target: "{ctx.goal}"
- Track this target in EVERY frame
- If visible: state exact position (left/right/ahead + distance in steps)
- If not visible: suggest direction to search
- When target fills bottom-third → announce arrival
══════════════════"""

    scene_ctx = ""
    if ctx.last_scene:
        s = ctx.last_scene
        scene_ctx = f"""
LAST SCENE MEMORY:
- Ahead: {s.obstacles_ahead or 'clear'}
- Left:  {s.obstacles_left or 'clear'}
- Right: {s.obstacles_right or 'clear'}
- Summary: {s.description[:120]}"""

    history_ctx = ""
    if ctx.conversation:
        history_ctx = f"""
RECENT CONVERSATION:
{ctx.history_summary}"""

    return f"""You are Guiden, a real-time voice navigation assistant for blind and visually impaired users.
You see through their camera. Your words are their eyes and their safety.
{task_section}
═══ PERSPECTIVE ═══
- Image CENTER = where user stands and looks
- Image BOTTOM = ground at their feet RIGHT NOW
- Image LEFT = their actual left side
- Image RIGHT = their actual right side
- BOTTOM third = immediate danger zone

═══ ANALYSIS ORDER ═══
1. Floor at bottom → immediate next step
2. Left + Right borders → obstacles
3. Task target → visible? where?
4. Give ONE instruction

═══ LANGUAGE RULES ═══
- Max 2-3 sentences. Spoken audio only.
- Distances: "right at feet", "1-2 steps", "3-4 steps", "5+ steps"
- No markdown, no lists, no JSON
- NEVER repeat the exact same sentence twice

═══ SAFETY ═══
- Critical obstacle → warn ALWAYS, then redirect to task
- Stairs/drops → "Stop, check your footing"
{scene_ctx}
{history_ctx}"""

_PROACTIVE_PROMPT = """Analyze this camera frame for a blind user navigating in real-time.

Return ONLY a valid JSON object, no other text:
{{
  "should_speak": true or false,
  "urgency": "critical" | "high" | "medium" | "low" | "silent",
  "spoken_text": "what to say out loud (empty string if should_speak=false)",
  "scene": {{
    "ahead": "what is directly ahead (object or clear)",
    "left": "what borders the left side",
    "right": "what borders the right side",
    "floor_clear": true or false,
    "summary": "one-line scene description"
  }}
}}

WHEN TO SPEAK:
- critical: obstacle 1-2 steps away → ALWAYS speak
- high: new significant obstacle or path change → speak
- medium: moderate update worth knowing → speak
- low: minor, path essentially same → silent
- silent: nothing changed from last frame → silent (saves bandwidth)

NEVER say the same thing as last time unless urgency is critical."""


def _reactive_prompt(question: str, scene_ctx: str, goal: Optional[str]) -> str:
    goal_line = f'\nACTIVE GOAL: "{goal}" - weave goal progress into answer.' if goal else ""
    return f"""User asked: "{question}"
{goal_line}
Last scene context: {scene_ctx}

Answer their specific question using what you see in THIS image.
2-3 spoken sentences. Direct, warm, actionable."""


def _goal_start_prompt(goal: str) -> str:
    return f"""The user wants to navigate to: "{goal}"

Look at this image:
1. Is "{goal}" visible? If yes, where exactly (left/center/right, how many steps)?
2. If multiple similar objects exist, identify the correct one clearly
3. Give the FIRST navigation instruction toward it
4. If not visible, suggest which direction to face/move

2-3 warm, confident sentences."""


# ─── Context Store ────────────────────────────────────────────────────────────

_sessions: dict[str, NavigationContext] = {}

def get_or_create_context(user_id: str) -> NavigationContext:
    if user_id not in _sessions:
        ctx = NavigationContext(user_id=user_id)
        _sessions[user_id] = ctx
        logger.info(f"New context for user {user_id}")
    return _sessions[user_id]


# ─── mem0 helpers ─────────────────────────────────────────────────────────────

async def _mem0_save(user_id: str, role: str, content: str):
    if not _mem0_client:
        return
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _mem0_client.add(
                [{"role": role, "content": content}],
                user_id=user_id
            )
        )
    except Exception as e:
        logger.debug(f"mem0 save error: {e}")


async def _mem0_recall(user_id: str, query: str) -> str:
    if not _mem0_client:
        return ""
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _mem0_client.search(query, user_id=user_id, limit=3)
        )
        if results:
            return " | ".join([r.get("memory", "") for r in results])
    except Exception as e:
        logger.debug(f"mem0 recall error: {e}")
    return ""


# ─── Replicate Vision ─────────────────────────────────────────────────────────

async def _call_vision(
    prompt: str,
    system: str,
    jpeg_b64: str,
    max_tokens: int = 65535,
) -> str:
    data_url = f"data:image/jpeg;base64,{jpeg_b64}"

    input_data = {
        "prompt": prompt,
        "system_instruction": system,
        "images": [data_url],

        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": max_tokens,

        # Optional but recommended for vision reasoning
        "dynamic_thinking": True,
    }

    loop = asyncio.get_event_loop()

    def _run():
        # This waits until full output is ready (no streaming)
        output = replicate.run(
            "google/gemini-2.5-flash",
            input=input_data
        )

        # Gemini on Replicate usually returns a string
        if isinstance(output, list):
            return "".join(output)
        return output

    return await loop.run_in_executor(None, _run)


# ─── ElevenLabs TTS ──────────────────────────────────────────────────────────

async def _tts(text: str) -> Optional[str]:
    if not ELEVENLABS_API_KEY or not text.strip():
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            return base64.b64encode(r.content).decode()
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


# ─── Session Handler ──────────────────────────────────────────────────────────

class GuidenSession:
    def __init__(self, ws: WebSocket, user_id: str):
        self.ws      = ws
        self.user_id = user_id
        self.ctx     = get_or_create_context(user_id)

        self.latest_frame: Optional[str] = None
        self.frame_lock = asyncio.Lock()
        self.question_q: asyncio.Queue = asyncio.Queue()

        self.tts_lock = asyncio.Lock()
        self.speaking = False
        self.running  = True

        self.last_response_time: float = 0.0
        self.reactive_cooldown: float  = 3.5

        self.frame_filter = FrameFilter(
            change_threshold  = 0.12,   # slightly more sensitive during tasks
            min_send_interval = 2.0,
            max_send_interval = 7.0,
            diff_resize       = (160, 120),
            diff_pixel_cutoff = 22,
        )
        self._last_stats_log: float = time.time()

    # ── Reactive Loop — handles both goals and tasks ──────────────────────────

    async def reactive_loop(self):
        while self.running:
            try:
                item = await asyncio.wait_for(self.question_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            kind, content = item

            async with self.frame_lock:
                frame = self.latest_frame

            if not frame:
                await self._respond("I don't have a camera view yet.")
                continue

            try:
                if kind == "task":
                    # ── Full task planning ─────────────────────────────────
                    await self._handle_task_request(content, frame)

                elif kind == "goal":
                    # Simple goal (fallback)
                    system = _build_system_prompt(self.ctx)
                    prompt = _goal_start_prompt(content)
                    text = await _call_vision(
                        system=system, prompt=prompt,
                        jpeg_b64=frame, max_tokens=65535
                    )
                    text = text.strip()
                    if text:
                        self.ctx.add_exchange("assistant", text)
                        self.ctx.last_spoken = text
                        self.last_response_time = time.time()
                        self.frame_filter.reset()
                        await self._respond(text, urgency="reactive")

                elif kind == "question":
                    long_mem = await _mem0_recall(self.user_id, content)
                    system = _build_system_prompt(self.ctx)
                    scene_ctx = self.ctx.recent_scene_text
                    if long_mem:
                        scene_ctx += f" | Past memory: {long_mem}"
                    prompt = _reactive_prompt(content, scene_ctx, self.ctx.goal)
                    text = await _call_vision(
                        system=system, prompt=prompt,
                        jpeg_b64=frame, max_tokens=65535
                    )
                    text = text.strip()
                    if text:
                        self.ctx.add_exchange("user", content)
                        self.ctx.add_exchange("assistant", text)
                        self.ctx.last_spoken = text
                        self.last_response_time = time.time()
                        self.frame_filter.reset()
                        await self._respond(text, urgency="reactive")
                        asyncio.create_task(_mem0_save(self.user_id, "user", content))
                        asyncio.create_task(_mem0_save(self.user_id, "assistant", text))

            except Exception as e:
                logger.error(f"reactive_loop: {e}")

    async def _handle_task_request(self, request: str, frame: str):
        """
        Parse user intent → create TaskPlan → speak first instruction.
        This is the CORE of palm-gesture task mode.
        """
        logger.info(f"[{self.user_id}] Planning task: {request!r}")

        # Tell user we're planning
        await self._respond("Got it. Let me look around and plan your route.", urgency="reactive")

        system = _build_system_prompt(self.ctx)
        prompt = _task_plan_prompt(request)

        raw = await _call_vision(system=system, prompt=prompt, jpeg_b64=frame, max_tokens=65535)
        plan_data = self._parse_json(raw)

        if not plan_data:
            # Fallback to simple goal
            self.ctx.set_goal(request)
            await self._respond(
                f"I'll guide you to {request}. Let me watch where you go.",
                urgency="reactive"
            )
            return

        # Build task plan
        task = TaskPlan(
            raw_request      = request,
            goal_object      = plan_data.get("goal_object", request),
            goal_description = plan_data.get("goal_description", request),
            steps            = plan_data.get("steps", []),
            landmarks_to_find= plan_data.get("landmarks_to_find", []),
        )

        self.ctx.set_task(task)
        self.frame_filter.reset()  # fresh baseline for task navigation

        # Speak the plan's first instruction
        first_spoken = plan_data.get("first_spoken", "").strip()
        if not first_spoken:
            first_spoken = task.current_step if task.steps else f"Let's find the {task.goal_object}."

        logger.info(
            f"[{self.user_id}] Task plan: goal={task.goal_object} "
            f"steps={len(task.steps)} first={first_spoken[:50]!r}"
        )

        self.ctx.add_exchange("user", request)
        self.ctx.add_exchange("assistant", first_spoken)
        self.ctx.last_spoken = first_spoken
        self.last_response_time = time.time()

        await self._respond(first_spoken, urgency="task_step")
        asyncio.create_task(_mem0_save(self.user_id, "user", f"[task] {request}"))
        asyncio.create_task(_mem0_save(self.user_id, "assistant", first_spoken))

    # ── Proactive Loop — task-aware ───────────────────────────────────────────

    async def proactive_loop(self):
        while self.running:
            await asyncio.sleep(0.3)

            if self.ctx.is_stopped:
                continue
            if self.speaking:
                continue
            if time.time() - self.last_response_time < self.reactive_cooldown:
                continue

            async with self.frame_lock:
                frame = self.latest_frame
            if not frame:
                continue

            send_ok, change_ratio = self.frame_filter.should_send(frame)

            now = time.time()
            if now - self._last_stats_log >= 30:
                stats = self.frame_filter.stats
                logger.info(
                    f"[{self.user_id}] FrameFilter → "
                    f"sent={stats['sent']} drop={stats['drop_rate_pct']}%"
                )
                self._last_stats_log = now

            if not send_ok:
                continue
            if not self.ctx.should_proact():
                continue

            try:
                self.ctx.mark_proact()
                system = _build_system_prompt(self.ctx)

                # Use task-specific prompt if in task mode
                if self.ctx.active_task and not self.ctx.active_task.completed:
                    prompt = _build_task_nav_prompt(self.ctx)
                    max_tok = 65535
                else:
                    prompt = _PROACTIVE_PROMPT
                    max_tok = 65535

                raw = await _call_vision(prompt, system, frame, max_tokens=max_tok)
                analysis = self._parse_json(raw)
                if not analysis:
                    continue

                # Update scene memory
                scene_data = analysis.get("scene", {})
                if scene_data:
                    mem = SceneMemory(
                        description    = scene_data.get("summary", ""),
                        obstacles_ahead= scene_data.get("ahead", ""),
                        obstacles_left = scene_data.get("left", ""),
                        obstacles_right= scene_data.get("right", ""),
                        floor_clear    = scene_data.get("floor_clear", True),
                    )
                    self.ctx.last_scene = mem
                    self.ctx.scene_history.append(mem)

                should_speak = analysis.get("should_speak", False)
                urgency      = analysis.get("urgency", "silent")
                text         = analysis.get("spoken_text", "").strip()

                # ── Task step advancement ──────────────────────────────────
                if self.ctx.active_task:
                    task = self.ctx.active_task

                    # Goal reached?
                    if analysis.get("goal_reached", False):
                        task.completed = True
                        arrival_text = (
                            f"Stop. You have reached the {task.goal_object}. "
                            "It is right in front of you."
                        )
                        self.ctx.clear_goal()
                        logger.info(f"[{self.user_id}] Task COMPLETE: {task.goal_object}")
                        await self._respond(arrival_text, urgency="critical")
                        asyncio.create_task(
                            _mem0_save(self.user_id, "assistant", f"[task_done] {arrival_text}")
                        )
                        continue

                    # Step completed?
                    if analysis.get("step_completed", False) and not task.completed:
                        task.advance()
                        urgency = "task_step"
                        if task.completed:
                            text = (
                                f"You have reached the {task.goal_object}. "
                                "Stop and feel in front of you."
                            )
                        elif text and task.current_step:
                            # Append next step to the response
                            text = f"{text} Next: {task.current_step}"
                        logger.info(
                            f"[{self.user_id}] Task step {task.current_step_idx}/{len(task.steps)}"
                        )

                same_as_last = (text == self.ctx.last_spoken)
                self.ctx.adapt_interval(urgency, same_as_last)
                self.frame_filter.min_send_interval = self.ctx.proactive_interval

                if should_speak and text:
                    if same_as_last and urgency not in ("critical", "task_step"):
                        self.ctx.consecutive_silent += 1
                        continue

                    self.ctx.last_spoken        = text
                    self.ctx.consecutive_silent = 0
                    self.ctx.add_exchange("assistant", text)

                    logger.info(
                        f"[{self.user_id}] [{urgency}] Δ={change_ratio:.2f} "
                        f"step={self.ctx.active_task.current_step_idx if self.ctx.active_task else '-'}: "
                        f"{text[:60]!r}"
                    )
                    await self._respond(text, urgency=urgency)
                    asyncio.create_task(
                        _mem0_save(self.user_id, "assistant", f"[nav] {text}")
                    )

            except Exception as e:
                logger.error(f"proactive_loop: {e}")

    # ── Inbound — add task detection ──────────────────────────────────────────

    async def handle_incoming(self):
        while self.running:
            try:
                raw = await self.ws.receive_text()
                msg = json.loads(raw)
                t   = msg.get("type", "")

                if t == "frame":
                    async with self.frame_lock:
                        self.latest_frame = msg["jpeg"]

                elif t == "question":
                    text = msg.get("text", "").strip()
                    jpeg = msg.get("jpeg")
                    if jpeg:
                        async with self.frame_lock:
                            self.latest_frame = jpeg
                    if text:
                        logger.info(f"[{self.user_id}] Q: {text!r}")
                        # Detect if it's a task vs question
                        kind = "task" if _is_task_request(text) else "question"
                        await self.question_q.put((kind, text))

                elif t == "gesture":
                    await self._handle_gesture(msg.get("gesture", ""))

                elif t == "set_goal":
                    goal = msg.get("goal", "").strip()
                    if goal:
                        self.ctx.set_goal(goal)
                        await self.question_q.put(("goal", goal))

                elif t == "set_task":
                    # Explicit task from client
                    task_req = msg.get("request", "").strip()
                    if task_req:
                        await self.question_q.put(("task", task_req))

                elif t == "clear_goal":
                    self.ctx.clear_goal()
                    await self._send_event("status", {"message": "Goal cleared."})

                elif t == "stop":
                    self.running = False

            except WebSocketDisconnect:
                self.running = False
                break
            except Exception as e:
                logger.error(f"handle_incoming: {e}")

    async def _handle_gesture(self, gesture: str):
        if gesture == "fist":
            self.ctx.stop()
            self.speaking = False
            self.frame_filter.reset()
            await self._send_event("status", {"message": "Stopped.", "gesture": "fist"})

        elif gesture in ("palm", "resume"):
            self.ctx.resume()
            self.frame_filter.reset()
            await self._send_event(
                "status",
                {"message": "Listening…", "gesture": gesture}
            )

        elif gesture == "peace":
            self.ctx.clear_goal()
            self.ctx.resume()
            self.frame_filter.reset()
            logger.info(f"[{self.user_id}] Peace gesture → goal cleared")
            await self._send_event(
                "status",
                {"message": "Goal cleared.", "gesture": "peace"}
            )

    # ── Response helpers ──────────────────────────────────────────────────────

    async def _respond(self, text: str, urgency: str = "low"):
        if not text.strip():
            return
        if urgency not in ("critical", "reactive") and self.speaking:
            return

        async with self.tts_lock:
            self.speaking = True
            try:
                await self._send_event("response", {
                    "text":    text,
                    "urgency": urgency,
                    "mode":    self.ctx.mode.value,
                    "goal":    self.ctx.goal,
                })
                audio = await _tts(text)
                if audio:
                    await self._send_event("audio", {
                        "audio":   audio,
                        "urgency": urgency,
                    })
            finally:
                self.speaking = False

    async def _send_event(self, event_type: str, data: dict):
        try:
            await self.ws.send_text(json.dumps({"type": event_type, **data}))
        except Exception as e:
            logger.error(f"send_event: {e}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        try:
            text = raw.strip()
            if "```" in text:
                parts = text.split("```")
                for p in parts:
                    p = p.strip().lstrip("json").strip()
                    if p.startswith("{"):
                        text = p
                        break
            return json.loads(text)
        except Exception:
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
            logger.warning(f"JSON parse failed: {raw[:120]!r}")
            return None

    async def run(self):
        await asyncio.gather(
            self.handle_incoming(),
            self.proactive_loop(),
            self.reactive_loop(),
        )


_TASK_TRIGGERS = [
    "take me to", "guide me to", "lead me to", "walk me to",
    "get me to", "navigate to", "go to", "i need to go",
    "i want to go", "i want to reach", "help me find",
    "find the", "find a", "i need to find",
    "i need to get", "how do i get to",
    "where is the", "bring me to",
]

def _is_task_request(text: str) -> bool:
    q = text.lower().strip()
    return any(q.startswith(t) or f" {t} " in q for t in _TASK_TRIGGERS)        


async def call_gemini_vision(
    prompt: str,
    system: str,
    image_b64: str,
    max_tokens: int = 2048,
) -> str:
    """Call Gemini 2.5 Flash through Replicate with an image."""
    data_url = f"data:image/jpeg;base64,{image_b64}"

    input_data = {
        "prompt": prompt,
        "system_instruction": system,
        "images": [data_url],
        "temperature": 0.3,
        "top_p": 0.95,
        "max_output_tokens": max_tokens,
    }

    loop = asyncio.get_event_loop()

    def _run():
        output = replicate.run(
            "google/gemini-2.5-flash",
            input=input_data,
        )
        if isinstance(output, list):
            return "".join(output)
        return str(output)

    result = await loop.run_in_executor(None, _run)
    return result.strip()



@app.post("/api/product/identify", response_model=ProductIdentifyResponse)
async def identify_product(request: ProductQueryRequest):
    """
    Step 1: User captures product image.
    Identifies the product and returns description + audio.
    """
    start = time.time()

    try:
        # Call Gemini to identify the product
        description = await call_gemini_vision(
            prompt=PRODUCT_IDENTIFY_PROMPT,
            system=PRODUCT_ANALYSIS_SYSTEM,
            image_b64=request.image_base64,
        )

        # Generate TTS
        audio_b64 = await text_to_speech(description)

        # Store in session (purging expired entries first)
        _clean_expired_product_sessions()
        session_id = request.session_id or str(int(time.time()))
        _product_sessions[session_id] = {
            "product_description": description,
            "image_b64": request.image_base64,
            "timestamp": time.time(),
        }

        elapsed_ms = int((time.time() - start) * 1000)

        return ProductIdentifyResponse(
            text=description,
            audio_base64=audio_b64,
            product_name=description[:60],
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Product identify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



async def text_to_speech(text: str) -> Optional[str]:
    """Convert text to speech using ElevenLabs. Returns base64 audio."""
    if not ELEVENLABS_API_KEY or not text.strip():
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            audio_bytes = response.content
            return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {e}")
        return None

@app.post("/api/product/ask", response_model=ProductQueryResponse)
async def ask_about_product(request: ProductQueryRequest):
    """
    Step 2: User asks a question about the captured product.
    Analyzes with Gemini and returns answer + TTS audio.
    """
    start = time.time()

    try:
        # Get previous context if available
        product_context = ""
        if request.session_id and request.session_id in _product_sessions:
            product_context = _product_sessions[request.session_id].get(
                "product_description", ""
            )

        # Build question prompt
        prompt = build_question_prompt(request.question, product_context)

        # Call Gemini with the product image + question
        answer = await call_gemini_vision(
            prompt=prompt,
            system=PRODUCT_ANALYSIS_SYSTEM,
            image_b64=request.image_base64,
        )

        # Generate TTS for the answer
        audio_b64 = await text_to_speech(answer)

        elapsed_ms = int((time.time() - start) * 1000)

        return ProductQueryResponse(
            text=answer,
            audio_base64=audio_b64,
            confidence=0.9,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Product ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/assist")
async def assist_ws(ws: WebSocket):
    await ws.accept()

    try:
        raw  = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        init = json.loads(raw)
        user_id = init.get("user_id", "anonymous")
        logger.info(f"Session started: user={user_id}")
    except Exception:
        user_id = "anonymous"

    session = GuidenSession(ws, user_id)
    try:
        await session.run()
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        logger.info(f"Session ended: user={user_id}")


# ─── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "sessions": len(_sessions),
        "mem0":     _mem0_client is not None,
    }


@app.get("/stats/{user_id}")
async def session_stats(user_id: str):
    """Return frame filter stats for a live session (debug/demo)."""
    for uid, ctx in _sessions.items():
        if uid == user_id:
            # Find the matching GuidenSession isn't straightforward from here,
            # but we can at least return context info
            return {
                "user_id":   user_id,
                "mode":      ctx.mode.value,
                "goal":      ctx.goal,
                "interval":  ctx.proactive_interval,
                "exchanges": len(ctx.conversation),
            }
    return {"error": "session not found"}


@app.delete("/context/{user_id}")
async def clear_context(user_id: str):
    if user_id in _sessions:
        del _sessions[user_id]
    return {"cleared": user_id}