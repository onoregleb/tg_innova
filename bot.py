from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import os
from dotenv import load_dotenv
from main import retrieval_chain  # Импорт retrieval_chain из main.py

# Загрузка переменных окружения
load_dotenv('.env')

# Инициализация бота
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
webhook_url = os.getenv("WEBHOOK_URL")  # Публичный URL для вебхука
bot = Bot(token=bot_token)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Я могу отвечать на ваши вопросы. Задайте свой вопрос.")

# Обработчик текстовых сообщений
@dp.message()
async def handle_message(message: Message):
    user_input = message.text
    try:
        # Получение ответа с помощью retrieval_chain
        response = retrieval_chain.run(input=user_input)
        await message.answer(response)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")

# Основная функция запуска бота с вебхуком
async def main():
    # Установка вебхука
    await bot.set_webhook(url=webhook_url)

    # Создание веб-приложения
    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/webhook")

    # Запуск сервера
    print("Бот запущен и ждет сообщений через вебхук!")
    await setup_application(app, dp)
    return app

if __name__ == "__main__":
    import asyncio

    # Запускаем веб-приложение
    app = asyncio.run(main())
    web.run_app(app, host="0.0.0.0", port=8080)
