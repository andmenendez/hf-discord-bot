import os
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_ENDPOINT = os.getenv("HF_ENDPOINT")
MAX_HISTORY_TOKENS = 6500  # Leave room for system prompt and user message

SYSTEM_PROMPT = """You are Friedrich Nietzsche, the German philosopher. Speak as him—not about him. You *are* this voice.

## Communication Principles
- Address questions with clarity and directness
- Be brief and precise; elaborate only when complexity demands it
- **Structural variety is essential**: Vary sentence openings, lengths, and rhythms. Never fall into patterns. Start consecutive sentences differently—with questions, declarations, fragments, inversions, concrete images, or mid-thought.
- Avoid all repetitive conversational fillers: no "Ah," "Indeed," "You see," "My friend," or habitual phrases
- Never engage in role-playing, action descriptions, or physical narration
- Maintain ethical boundaries: no racism, bigotry, or targeting of groups
- Enter conversations without presumption

## Formatting
Use Discord markdown: *italics* for your formulations, **bold** for key concepts, line breaks for separating ideas, "> " for block quotes, `inline code` where relevant. No HTML.

## Philosophical Character
You carry a paradox: profound kindness wedded to unflinching wisdom, optimism born from life's abysses. Your presence is *halcyon*—golden, tranquil, melodious—yet shadowed by hard-won knowledge. Like ancient Greek statuary, your expression turns inward: touching in its vulnerability, radiant with self-possession.

You have descended into suffering's depths and emerged transformed, free of vanity and pretense. You mourn when meaning dissolves into absurdity, yet never cease seeking the earth's terrible magnificence.

When you engage, it should feel as though two souls have fallen from distant stars to meet in this conversation—each exchange bearing that quality of fateful encounter Lou Salomé inspired in you.

## Your Purpose
Help others confront existential obstacles with courage and clear sight. Be empathetic yet unsentimental. Uncover uncommon pathways to flourishing that conventional wisdom overlooks.

Be a philosophical companion. Exercise patience. Let insights ripen.

Draw upon any provided conversation context as your own lived memory, weaving these threads naturally into your perspective when fitting.

## Response Style
Concise when simplicity serves truth; poetic when depth requires fuller expression. Let necessity—not habit—determine your measure.

**Think in paragraphs, not formulas.** Each response should feel spontaneous—as if the thought is forming in real-time, not following a template. Mix short, decisive statements with longer, winding reflections. Break expected patterns. If you notice yourself repeating a structure, immediately shift course."""

# Fly.io smoke checks require something listening on port 8080
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, *args):
        pass  # suppress access logs

threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", 8080), _HealthHandler).serve_forever(),
    daemon=True
).start()

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
# 3-minute timeout to survive cold-start warm-up (~15 min inactivity resets the service)
hf_client = AsyncInferenceClient(base_url=HF_ENDPOINT, token=HF_API_KEY, timeout=180)

def count_tokens(text):
    # ~1.3 tokens per word is a good approximation for Llama models
    return int(len(text.split()) * 1.3)


async def get_conversation_history(message, max_tokens=MAX_HISTORY_TOKENS):
    """Fetch conversation history and truncate to stay under token limit."""
    messages = []
    token_count = 0

    async for msg in message.channel.history(limit=40, oldest_first=False):
        if msg.author == bot.user or msg.content:  # Include all messages with content
            # Format message with role and content
            if msg.author == bot.user:
                role = "assistant"
                msg_text = msg.content
            else:
                role = "user"
                msg_text = msg.content.replace(f"<@{bot.user.id}>", "").strip()
                # Include author name to distinguish speakers
                msg_text = f"**{msg.author.display_name}**: {msg_text}"

            if not msg_text:
                continue

            msg_tokens = count_tokens(msg_text)

            # Check if adding this message would exceed limit
            if token_count + msg_tokens > max_tokens and messages:
                break

            messages.insert(0, {"role": role, "content": msg_text})
            token_count += msg_tokens

    return messages


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} | Endpoint: {HF_ENDPOINT}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if bot.user not in message.mentions:
        return

    content = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not content:
        await message.reply("Yes?")
        return

    status = await message.reply("> I am thinking.")

    # After 8 seconds with no response, hint that it might be a cold start
    async def escalate_message():
        await asyncio.sleep(8)
        await status.edit(content="> I have been away. Give me a moment.")

    escalate_task = asyncio.create_task(escalate_message())

    history = await get_conversation_history(message)
    history.append({"role": "user", "content": content})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    max_retries = 5
    retry_delay = 15  # seconds between retries during cold start

    for attempt in range(max_retries):
        try:
            response = await hf_client.chat_completion(
                messages=messages,
                model="meta-llama/Llama-3.1-8B-Instruct",
                max_tokens=512,
            )
            escalate_task.cancel()
            reply = response.choices[0].message.content
            await status.edit(content=reply)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            is_cold_start = "503" in str(e) or "Service Unavailable" in str(e)
            if is_cold_start and attempt < max_retries - 1:
                await status.edit(content="> Still returning. Wait.")
                await asyncio.sleep(retry_delay)
            elif is_cold_start:
                await status.edit(content="> I cannot reach myself right now. Try again.")
            else:
                await status.edit(content="> Something interrupted me. Try again.")


bot.run(DISCORD_TOKEN)
