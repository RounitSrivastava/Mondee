import os
from dotenv import load_dotenv
from groq import Groq

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters
)

from tools import calculator, web_search

# Load .env file
load_dotenv()

# Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_message = update.message.text.lower()

    # Calculator Tool
    if user_message.startswith("calculate"):

        expression = user_message.replace("calculate", "")

        result = calculator(expression)

        await update.message.reply_text(f"🧮 Result: {result}")

        return

    # Search Tool
    elif user_message.startswith("search"):

        query = user_message.replace("search", "")

        result = web_search(query)

        await update.message.reply_text(result)

        return

    # AI Response
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": "You are a powerful agentic AI assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        temperature=0.7,
        max_tokens=1024
    )

    ai_reply = completion.choices[0].message.content

    await update.message.reply_text(ai_reply)

# Start Telegram Bot
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT, handle_message))

print("🚀 Bot Running...")

app.run_polling()