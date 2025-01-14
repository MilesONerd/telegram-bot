import asyncio
import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from ai_handler import ai_handler  # Import the AI handler singleton

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm MilesONerd AI, your intelligent assistant.\n"
        "I can help you with various tasks using advanced AI models and internet search.\n"
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
Available commands:
/start - Start the bot
/help - Show this help message
/about - Learn more about MilesONerd AI

Message handling:
- Short questions: Quick responses using lightweight model
- Long messages: Summarization and detailed response
- Include 'summarize' or 'tldr' for text summarization
- Chat-related queries: Optimized conversation handling
- Regular messages: Comprehensive AI-powered responses

You can send me any message, and I'll process it using the most appropriate AI model!
    """
    await update.message.reply_text(help_text)

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send information about the bot when the command /about is issued."""
    about_text = """
MilesONerd AI is an intelligent assistant powered by advanced AI models and internet search capabilities.

Features:
- Advanced language understanding
- Internet search integration
- Continuous learning from interactions
- Multiple AI models for different tasks

Created with ❤️ using python-telegram-bot and Hugging Face models.
    """
    await update.message.reply_text(about_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages using appropriate AI models based on content."""
    try:
        user_message = update.message.text.strip()
        
        # Send typing action while processing
        await update.message.chat.send_action(action="typing")
        
        # Check message length and type to determine appropriate model
        if len(user_message.split()) > 100:  # Long messages
            # Use BART for summarization first
            summary = await ai_handler.summarize_text(user_message)
            # Then use GPT-2 for response generation
            response = await ai_handler.generate_response(
                f"Based on this summary: {summary}\nGenerate a helpful response:",
                model_key='gpt2',
                max_length=200
            )
        elif any(keyword in user_message.lower() for keyword in ['summarize', 'summary', 'tldr']):
            # Explicit summarization request
            response = await ai_handler.summarize_text(user_message)
        # Use GPT-2 for conversational queries
        elif any(keyword in user_message.lower() for keyword in ['chat', 'conversation', 'talk']):
            response = await ai_handler.generate_response(
                user_message,
                model_key='gpt2',
                max_length=200
            )
        elif len(user_message.split()) < 10:  # Short queries
            # Use GPT-2 for quick responses to short queries
            response = await ai_handler.generate_response(
                user_message,
                model_key='gpt2',
                max_length=100
            )
        else:
            # Default to GPT-2 for general responses
            response = await ai_handler.generate_response(
                user_message,
                model_key='gpt2',
                max_length=150
            )
        
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await update.message.reply_text(
            "I apologize, but I encountered an error while processing your message. "
            "Please try again later."
        )

async def initialize() -> bool:
    """Initialize AI models and other components."""
    try:
        logger.info("Initializing AI models...")
        success = await ai_handler.initialize_models()
        if not success:
            logger.error("Failed to initialize AI models")
            return False
        logger.info("AI models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        return False

def main() -> None:
    """Start the bot."""
    # Get the token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("No token found! Make sure to set TELEGRAM_BOT_TOKEN in .env file")
        return

    # Create the Application and initialize models
    application = Application.builder().token(token).build()
    
    # Initialize models before starting the bot
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run initialization in the event loop
        init_success = loop.run_until_complete(initialize())
        if not init_success:
            logger.error("Failed to initialize AI models. Exiting...")
            return
        logger.info("AI models initialized successfully")
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("about", about_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Log startup
        logger.info("MilesONerd AI Bot is starting...")
        
        # Run the bot
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        loop.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
        raise  # Re-raise the exception for proper error handling
