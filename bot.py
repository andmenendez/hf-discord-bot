import os
import discord
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
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

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
# 3-minute timeout to survive cold-start warm-up (~15 min inactivity resets the service)
hf_client = AsyncInferenceClient(base_url=HF_ENDPOINT, token=HF_API_KEY, timeout=180)

# Load tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


def count_tokens(text):
    """Count tokens in text using Llama tokenizer."""
    return len(tokenizer.encode(text))


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

    # Acknowledge immediately so the user knows something is happening.
    # If the service is cold, this may sit for 30-60 seconds before editing.
    status = await message.reply("Waking up... give me a moment ☕")

    try:
        # Get conversation history with token limiting
        history = await get_conversation_history(message)
        # Append current user message
        history.append({"role": "user", "content": content})

        # Prepend system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        response = await hf_client.chat_completion(
            messages=messages,
            max_tokens=512,
        )
        reply = response.choices[0].message.content
        await status.edit(content=reply)
    except Exception as e:
        print(f"Error: {e}")
        await status.edit(content=f"Something went wrong: {e}")


bot.run(DISCORD_TOKEN)
