from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
import os
from dotenv import load_dotenv
from main import retrieval_chain

# Загрузка переменных окружения
load_dotenv('.env')

# Инициализация бота
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
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
        response = retrieval_chain.invoke(
            {'input': user_input}
        )
        # Убедимся, что response — это строка
        if isinstance(response, dict):
            print(response)
            response = response.get('answer', 'Нет данных для ответа')
        await message.answer(str(response))
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")


async def main():
    # Запуск бота в режиме polling
    print("Бот запущен и ждет сообщений!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    # Запуск бота
    asyncio.run(main())
