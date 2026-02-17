import asyncio
import html
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import feedparser
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatMemberStatus, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from deep_translator import GoogleTranslator
from PIL import Image

from config import ADMIN_ID, BOT_TOKEN, CHANNEL_ID

RSS_SOURCES = [
    "https://www.ign.com/rss",
    "https://www.gamesradar.com/feeds/all",
    "https://www.pcgamer.com/rss/",
    "https://www.vg247.com/feed",
    "https://www.rockpapershotgun.com/feed",
    "https://www.gamespot.com/feeds/mashup/",
    "https://kotaku.com/rss",
]

STORAGE_FILE = Path("storage.json")

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="1️⃣ Новий пост")],
        [KeyboardButton(text="2️⃣ Опитування")],
        [KeyboardButton(text="3️⃣ Скасувати")],
    ],
    resize_keyboard=True,
)

CONFIRM_POST_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="✅ Опублікувати", callback_data="publish_post")],
        [InlineKeyboardButton(text="❌ Скасувати", callback_data="cancel_post")],
    ]
)

CONFIRM_POLL_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="✅ Опублікувати", callback_data="publish_poll")],
        [InlineKeyboardButton(text="❌ Скасувати", callback_data="cancel_poll")],
    ]
)


class BotStates(StatesGroup):
    idle = State()
    post_preview = State()
    poll_preview = State()


@dataclass
class DraftPost:
    link: str
    caption: str
    media_type: str
    media_bytes: bytes
    filename: str


@dataclass
class DraftPoll:
    question: str
    options: list[str]


POST_DRAFTS: dict[int, DraftPost] = {}
POLL_DRAFTS: dict[int, DraftPoll] = {}


class JsonStorage:
    def __init__(self, path: Path):
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.write_text(json.dumps({"published_links": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        self._ensure_file()
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if "published_links" not in data or not isinstance(data["published_links"], list):
            data = {"published_links": []}
        return data

    def is_published(self, link: str) -> bool:
        data = self.load()
        return link in data["published_links"]

    def add_published(self, link: str) -> None:
        data = self.load()
        if link not in data["published_links"]:
            data["published_links"].append(link)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


storage = JsonStorage(STORAGE_FILE)


def sanitize_html(text: str) -> str:
    return html.escape(text, quote=False)


def need_translation(text: str) -> bool:
    latin = sum(1 for ch in text.lower() if "a" <= ch <= "z")
    cyrillic = sum(1 for ch in text.lower() if "а" <= ch <= "я" or ch in "іїєґ")
    return latin > cyrillic


def translate_to_ua(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    if not need_translation(cleaned):
        return cleaned
    try:
        return GoogleTranslator(source="auto", target="uk").translate(cleaned)
    except Exception:
        return cleaned


def extract_media_urls(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []

    for media in entry.get("media_content", []):
        url = media.get("url", "")
        m_type = media.get("type", "")
        if not url:
            continue
        if "video" in m_type or url.lower().endswith((".mp4", ".mov", ".webm")):
            videos.append(url)
        elif "image" in m_type or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(url)

    for media in entry.get("media_thumbnail", []):
        url = media.get("url", "")
        if url:
            images.append(url)

    for link in entry.get("links", []):
        url = link.get("href", "")
        m_type = link.get("type", "")
        if not url:
            continue
        if "video" in m_type or url.lower().endswith((".mp4", ".mov", ".webm")):
            videos.append(url)
        elif "image" in m_type or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(url)

    enclosure = entry.get("enclosures", [])
    for item in enclosure:
        url = item.get("href", "")
        if not url:
            continue
        if url.lower().endswith((".mp4", ".mov", ".webm")):
            videos.append(url)
        elif url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(url)

    return list(dict.fromkeys(images)), list(dict.fromkeys(videos))


async def download_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
        response.raise_for_status()
        return await response.read()


def build_collage(images_data: list[bytes]) -> bytes:
    target_size = (1280, 720)
    collage = Image.new("RGB", target_size, color=(20, 20, 20))
    slots = [
        (0, 0, 426, 720),
        (426, 0, 854, 720),
        (854, 0, 1280, 720),
    ]

    for idx, image_data in enumerate(images_data[:3]):
        with Image.open(BytesIO(image_data)) as img:
            img = img.convert("RGB")
            left, top, right, bottom = slots[idx]
            slot_w = right - left
            slot_h = bottom - top
            ratio = max(slot_w / img.width, slot_h / img.height)
            resized = img.resize((int(img.width * ratio), int(img.height * ratio)))
            x = (resized.width - slot_w) // 2
            y = (resized.height - slot_h) // 2
            crop = resized.crop((x, y, x + slot_w, y + slot_h))
            collage.paste(crop, (left, top))

    out = BytesIO()
    collage.save(out, format="JPEG", quality=90)
    return out.getvalue()


async def find_news() -> tuple[str, str, str, list[str], list[str]]:
    async with aiohttp.ClientSession() as session:
        for feed_url in RSS_SOURCES:
            parsed = feedparser.parse(feed_url)
            entries = parsed.entries or []
            for entry in entries:
                link = entry.get("link", "").strip()
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "") or entry.get("description", "") or ""
                if not link or not title:
                    continue
                if storage.is_published(link):
                    continue

                images, videos = extract_media_urls(entry)
                clean_summary = " ".join(summary.replace("<br>", " ").replace("<p>", " ").replace("</p>", " ").split())
                text = translate_to_ua(clean_summary)
                translated_title = translate_to_ua(title)
                if not text:
                    text = "Деталі новини скоро у каналі."
                return link, translated_title, text, images, videos
    raise RuntimeError("Не вдалося знайти нову новину в RSS.")


def make_caption(title: str, text: str, game_link: str) -> str:
    short_lines = text.split(". ")
    body = ".\n".join(short_lines[:4]).strip()
    if len(body.splitlines()) < 2:
        body = text
    body = "\n".join(body.splitlines()[:10]).strip()
    game_anchor = f'<a href="{sanitize_html(game_link)}">посилання на гру</a>'
    return (
        f"<b>🎮 {sanitize_html(title)}</b>\n\n"
        f"{sanitize_html(body)}\n\n"
        f"Детальніше: {game_anchor}\n\n"
        "🔗 Темні Ігри | Підписуйся\n"
        "https://t.me/hhh_play"
    )


async def create_post_draft() -> DraftPost:
    link, title, text, images, videos = await find_news()
    caption = make_caption(title, text, link)

    async with aiohttp.ClientSession() as session:
        if videos:
            video_bytes = await download_bytes(session, videos[0])
            return DraftPost(link=link, caption=caption, media_type="video", media_bytes=video_bytes, filename="news.mp4")

        if images:
            first_image = await download_bytes(session, images[0])
            if len(images) >= 3:
                images_data: list[bytes] = []
                for image_url in images[:3]:
                    try:
                        images_data.append(await download_bytes(session, image_url))
                    except Exception:
                        continue
                if len(images_data) == 3:
                    collage_bytes = build_collage(images_data)
                    return DraftPost(
                        link=link,
                        caption=caption,
                        media_type="photo",
                        media_bytes=collage_bytes,
                        filename="collage.jpg",
                    )
            return DraftPost(link=link, caption=caption, media_type="photo", media_bytes=first_image, filename="news.jpg")

    raise RuntimeError("У новині немає медіа для публікації.")


def create_poll_draft() -> DraftPoll:
    return DraftPoll(
        question="🎮 Який жанр гри вам зараз найближчий?",
        options=["RPG", "Шутер", "Стратегія", "Інді"],
    )


async def ensure_access(message: Message) -> bool:
    if message.from_user is None or message.from_user.id != ADMIN_ID:
        await message.answer("Доступ заборонено.")
        return False
    return True


async def ensure_bot_is_admin(bot: Bot) -> bool:
    me = await bot.get_me()
    member = await bot.get_chat_member(chat_id=CHANNEL_ID, user_id=me.id)
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}


async def cmd_start(message: Message, state: FSMContext) -> None:
    if not await ensure_access(message):
        return
    await state.set_state(BotStates.idle)
    await message.answer("Оберіть дію:", reply_markup=MAIN_MENU)


async def create_new_post(message: Message, state: FSMContext, bot: Bot) -> None:
    if not await ensure_access(message):
        return
    if not await ensure_bot_is_admin(bot):
        await message.answer("Бот не є адміністратором каналу.")
        return

    await message.answer("Шукаю новину та формую пост...")
    draft = await create_post_draft()
    POST_DRAFTS[message.from_user.id] = draft

    media = BufferedInputFile(draft.media_bytes, filename=draft.filename)
    if draft.media_type == "video":
        await message.answer_video(video=media, caption=draft.caption, reply_markup=CONFIRM_POST_KB)
    else:
        await message.answer_photo(photo=media, caption=draft.caption, reply_markup=CONFIRM_POST_KB)

    await state.set_state(BotStates.post_preview)


async def create_poll(message: Message, state: FSMContext, bot: Bot) -> None:
    if not await ensure_access(message):
        return
    if not await ensure_bot_is_admin(bot):
        await message.answer("Бот не є адміністратором каналу.")
        return

    draft = create_poll_draft()
    POLL_DRAFTS[message.from_user.id] = draft

    await message.answer_poll(
        question=draft.question,
        options=draft.options,
        is_anonymous=False,
    )
    await message.answer("Опитування готове до публікації:", reply_markup=CONFIRM_POLL_KB)
    await state.set_state(BotStates.poll_preview)


async def cancel_action(message: Message, state: FSMContext) -> None:
    if not await ensure_access(message):
        return
    POST_DRAFTS.pop(message.from_user.id, None)
    POLL_DRAFTS.pop(message.from_user.id, None)
    await state.clear()
    await state.set_state(BotStates.idle)
    await message.answer("Скасовано. Повертаю в головне меню.", reply_markup=MAIN_MENU)


async def publish_post(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    user_id = callback.from_user.id
    draft = POST_DRAFTS.get(user_id)
    if draft is None:
        await callback.answer("Чернетку не знайдено.", show_alert=True)
        return

    if not await ensure_bot_is_admin(bot):
        await callback.answer("Бот не є адміністратором каналу.", show_alert=True)
        return

    media = BufferedInputFile(draft.media_bytes, filename=draft.filename)
    if draft.media_type == "video":
        await bot.send_video(chat_id=CHANNEL_ID, video=media, caption=draft.caption)
    else:
        await bot.send_photo(chat_id=CHANNEL_ID, photo=media, caption=draft.caption)

    storage.add_published(draft.link)
    POST_DRAFTS.pop(user_id, None)
    await state.set_state(BotStates.idle)
    await callback.message.answer("Пост опубліковано в канал.", reply_markup=MAIN_MENU)
    await callback.answer()


async def cancel_post(callback: CallbackQuery, state: FSMContext) -> None:
    POST_DRAFTS.pop(callback.from_user.id, None)
    await state.set_state(BotStates.idle)
    await callback.message.answer("Публікацію поста скасовано.", reply_markup=MAIN_MENU)
    await callback.answer()


async def publish_poll(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    user_id = callback.from_user.id
    draft = POLL_DRAFTS.get(user_id)
    if draft is None:
        await callback.answer("Чернетку опитування не знайдено.", show_alert=True)
        return

    if not await ensure_bot_is_admin(bot):
        await callback.answer("Бот не є адміністратором каналу.", show_alert=True)
        return

    try:
        await bot.send_poll(
            chat_id=CHANNEL_ID,
            question=draft.question,
            options=draft.options,
            is_anonymous=False,
        )
    except TelegramBadRequest as error:
        logging.exception("Помилка публікації опитування: %s", error)
        await callback.answer("Не вдалося опублікувати опитування.", show_alert=True)
        return

    POLL_DRAFTS.pop(user_id, None)
    await state.set_state(BotStates.idle)
    await callback.message.answer("Опитування опубліковано в канал.", reply_markup=MAIN_MENU)
    await callback.answer()


async def cancel_poll(callback: CallbackQuery, state: FSMContext) -> None:
    POLL_DRAFTS.pop(callback.from_user.id, None)
    await state.set_state(BotStates.idle)
    await callback.message.answer("Публікацію опитування скасовано.", reply_markup=MAIN_MENU)
    await callback.answer()


async def handle_errors(event: Any, exception: Exception) -> bool:
    logging.exception("Unhandled error: %s", exception)
    if isinstance(event, Message):
        await event.answer("Сталася помилка. Спробуйте ще раз.", reply_markup=MAIN_MENU)
    elif isinstance(event, CallbackQuery):
        await event.answer("Сталася помилка.", show_alert=True)
    return True


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не заданий")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID не заданий")
    if not ADMIN_ID:
        raise RuntimeError("ADMIN_ID не заданий")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(create_new_post, F.text == "1️⃣ Новий пост")
    dp.message.register(create_poll, F.text == "2️⃣ Опитування")
    dp.message.register(cancel_action, F.text == "3️⃣ Скасувати")

    dp.callback_query.register(publish_post, F.data == "publish_post")
    dp.callback_query.register(cancel_post, F.data == "cancel_post")
    dp.callback_query.register(publish_poll, F.data == "publish_poll")
    dp.callback_query.register(cancel_poll, F.data == "cancel_poll")

    dp.error.register(handle_errors)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
