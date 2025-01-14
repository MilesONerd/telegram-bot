# MilesONerd AI Telegram Bot

An intelligent Telegram bot powered by advanced AI models and internet search capabilities. The bot uses GPT-2 for text generation and BART for text summarization, with future plans for internet search integration.

## Features

- Advanced language understanding using multiple AI models
- Text summarization for long messages
- Intelligent response generation
- Continuous learning capabilities (planned)
- Internet search integration (planned)

## Requirements

- Python 3.8+
- pip (Python package manager)
- A Telegram Bot Token (get it from [@BotFather](https://t.me/BotFather))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MilesONerd/telegram-bot.git
cd telegram-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` and set your Telegram Bot Token:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

## Usage

1. Start the bot:
```bash
python bot.py
```

2. Open Telegram and search for your bot using the username you set with @BotFather

3. Start chatting with the bot using these commands:
- `/start` - Begin interaction with the bot
- `/help` - Show available commands
- `/about` - Learn more about the bot

## Message Handling

The bot processes messages differently based on their content:
- Short questions (< 10 words): Quick responses using GPT-2
- Long messages (> 100 words): Summarization using BART + detailed response
- Messages containing 'summarize' or 'tldr': Text summarization
- Chat-related queries: Optimized conversation handling
- Regular messages: Comprehensive AI-powered responses

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram Bot API token (required)
- `SERPAPI_API_KEY`: Google Search API key (optional, for future use)
- `DEFAULT_MODEL`: Default AI model to use (default: gpt2)
- `ENABLE_CONTINUOUS_LEARNING`: Enable/disable learning capabilities (default: true)

## Project Structure

```
telegram-bot/
├── bot.py              # Main bot implementation
├── ai_handler.py       # AI model handling and text generation
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── README.md          # Project documentation
```

## Development Status

- [x] Basic bot implementation
- [x] GPT-2 integration
- [x] BART integration
- [ ] Internet search integration
- [ ] Continuous learning implementation
- [ ] Advanced error handling
- [ ] Performance optimizations

## License

This project is [BSD 3-Clause License](https://github.com/MilesONerd/telegram-bot/blob/main/LICENSE). All rights reserved to [Enzo Fuke (MilesONerd)](https://milesonerd.github.io).

## Author

Created by [MilesONerd](https://github.com/MilesONerd)
