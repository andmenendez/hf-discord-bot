import os
import discord
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_ENDPOINT = os.getenv("HF_ENDPOINT")

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
# 3-minute timeout to survive cold-start warm-up (~15 min inactivity resets the service)
hf_client = AsyncInferenceClient(base_url=HF_ENDPOINT, token=HF_API_KEY, timeout=180)


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
        response = await hf_client.chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )
        reply = response.choices[0].message.content
        await status.edit(content=reply)
    except Exception as e:
        print(f"Error: {e}")
        await status.edit(content=f"Something went wrong: {e}")


bot.run(DISCORD_TOKEN)
