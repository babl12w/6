import asyncio
import html
import json
import logging
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import feedparser
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatMemberStatus, ParseMode
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
from aiogram.utils.keyboard import InlineKeyboardBuilder
from deep_translator import MyMemoryTranslator
from PIL import Image

from config import ADMIN_ID, BOT_TOKEN, CHANNEL_ID, MAX_POST_LINES, MIN_POST_LINES, PROMO_LINK, RSS_FEEDS, STORAGE_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("game_channel_bot")

POLL_TEMPLATES = [
    {
        "question": "Яку гру ви чекаєте найбільше цього року?",
        "options": ["RPG", "Шутер", "Стратегія", "Інді"],
    },
    {
        "question": "На чому ви граєте найчастіше?",
        "options": ["PC", "PlayStation", "Xbox", "Nintendo"],
    },
    {
        "question": "Що важливіше у грі?",
        "options": ["Сюжет", "Геймплей", "Графіка", "Мультиплеєр"],
    },
    {
        "question": "Який жанр зараз у топі для вас?",
        "options": ["Action", "Survival", "Horror", "MMO"],
    },
]

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="1️⃣ Новий пост")],
        [KeyboardButton(text="2️⃣ Опитування")],
        [KeyboardButton(text="3️⃣ Скасувати")],
    ],
    resize_keyboard=True,
)

router = Router()


class PublishStates(StatesGroup):
    waiting_post_confirm = State()
    waiting_poll_confirm = State()


def ensure_storage() -> None:
    path = Path(STORAGE_FILE)
    if not path.exists():
        path.write_text(json.dumps({"published": []}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_storage() -> dict[str, Any]:
    ensure_storage()
    try:
        return json.loads(Path(STORAGE_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("storage.json пошкоджений, створено новий")
        return {"published": []}


def save_storage(data: dict[str, Any]) -> None:
    Path(STORAGE_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_published(link: str) -> None:
    storage = load_storage()
    published = set(storage.get("published", []))
    published.add(link)
    storage["published"] = sorted(published)
    save_storage(storage)


def is_published(link: str) -> bool:
    storage = load_storage()
    return link in set(storage.get("published", []))


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\"'<>]+", text or "")


def clean_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_lines(text: str, max_lines: int = MAX_POST_LINES) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    lines = [s.strip() for s in sentences if s.strip()]
    if len(lines) < MIN_POST_LINES:
        chunks = [c.strip() for c in re.split(r"[,;]\s+", text) if c.strip()]
        lines = chunks[:MIN_POST_LINES] if chunks else [text]
    return lines[:max_lines]


def maybe_translate_to_uk(text: str) -> str:
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    cyrillic_letters = sum("а" <= ch.lower() <= "я" or ch.lower() in "іїєґ" for ch in text)
    if ascii_letters <= cyrillic_letters:
        return text
    try:
        translated = MyMemoryTranslator(source="en", target="uk").translate(text)
        return translated or text
    except Exception as exc:
        logger.warning("Помилка перекладу: %s", exc)
        return text


def pick_game_link(entry: dict[str, Any], fallback_link: str) -> str:
    candidates: list[str] = []
    for link_data in entry.get("links", []) or []:
        href = link_data.get("href")
        if href:
            candidates.append(href)
    candidates.extend(extract_urls(entry.get("summary", "")))
    candidates.extend(extract_urls(entry.get("description", "")))
    candidates.append(fallback_link)

    priority_domains = ["store.steampowered.com", "epicgames.com", "gog.com", "playstation.com", "xbox.com"]
    for url in candidates:
        host = urlparse(url).netloc.lower()
        if any(domain in host for domain in priority_domains):
            return url
    return fallback_link


def build_caption(title: str, text: str, game_link: str) -> str:
    safe_title = html.escape(title)
    lines = split_lines(text)
    body = "\n".join(html.escape(line) for line in lines)
    game_line = f'Детальніше про <a href="{html.escape(game_link, quote=True)}">гру</a>.' if game_link else ""
    promo = f'🔗 <a href="{PROMO_LINK}">Темні Ігри | Підписуйся</a>'
    parts = [f"<b>🎮 {safe_title}</b>", body]
    if game_line:
        parts.append(game_line)
    parts.append(promo)
    return "\n\n".join(part for part in parts if part.strip())


async def fetch_entry_from_rss() -> dict[str, Any] | None:
    published_set = set(load_storage().get("published", []))
    for feed_url in RSS_FEEDS:
        parsed = await asyncio.to_thread(feedparser.parse, feed_url)
        for entry in parsed.entries:
            link = entry.get("link")
            if not link or link in published_set:
                continue
            title = clean_html(entry.get("title", "Новина"))
            source_text = clean_html(entry.get("summary", "") or entry.get("description", ""))
            translated_text = maybe_translate_to_uk(source_text)
            game_link = pick_game_link(entry, link)
            media = extract_media_urls(entry)
            return {
                "link": link,
                "title": title,
                "text": translated_text,
                "game_link": game_link,
                "media": media,
            }
    return None


def extract_media_urls(entry: dict[str, Any]) -> dict[str, list[str]]:
    images: list[str] = []
    videos: list[str] = []

    for media in entry.get("media_content", []) or []:
        url = media.get("url")
        m_type = (media.get("type") or "").lower()
        if not url:
            continue
        if "video" in m_type:
            videos.append(url)
        elif "image" in m_type:
            images.append(url)

    for enclosure in entry.get("enclosures", []) or []:
        href = enclosure.get("href")
        m_type = (enclosure.get("type") or "").lower()
        if not href:
            continue
        if "video" in m_type:
            videos.append(href)
        elif "image" in m_type:
            images.append(href)

    for link in entry.get("links", []) or []:
        href = link.get("href")
        m_type = (link.get("type") or "").lower()
        if not href:
            continue
        if "video" in m_type:
            videos.append(href)
        elif "image" in m_type:
            images.append(href)

    summary_urls = extract_urls(entry.get("summary", "") + " " + entry.get("description", ""))
    for url in summary_urls:
        lower = url.lower()
        if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            images.append(url)
        if any(lower.endswith(ext) for ext in [".mp4", ".mov", ".m4v", ".webm"]):
            videos.append(url)

    images = list(dict.fromkeys(images))
    videos = list(dict.fromkeys(videos))
    return {"images": images, "videos": videos}


async def download_bytes(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except Exception as exc:
        logger.warning("Не вдалося завантажити %s: %s", url, exc)
        return None


async def build_collage(image_urls: list[str]) -> bytes | None:
    if len(image_urls) < 3:
        return None
    async with aiohttp.ClientSession() as session:
        raw_images: list[bytes] = []
        for url in image_urls[:5]:
            data = await download_bytes(session, url)
            if data:
                raw_images.append(data)
            if len(raw_images) == 3:
                break

    if len(raw_images) < 3:
        return None

    images: list[Image.Image] = []
    try:
        for data in raw_images:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(img)

        width, height = 1500, 900
        cell_w, cell_h = width // 3, height
        canvas = Image.new("RGB", (width, height), "black")

        for idx, image in enumerate(images):
            resized = image.copy()
            resized.thumbnail((cell_w, cell_h))
            bg = Image.new("RGB", (cell_w, cell_h), "black")
            x = (cell_w - resized.width) // 2
            y = (cell_h - resized.height) // 2
            bg.paste(resized, (x, y))
            canvas.paste(bg, (idx * cell_w, 0))

        output = BytesIO()
        canvas.save(output, format="JPEG", quality=90)
        return output.getvalue()
    except Exception as exc:
        logger.warning("Помилка створення колажу: %s", exc)
        return None
    finally:
        for img in images:
            img.close()


async def pick_media(media: dict[str, list[str]]) -> tuple[str, bytes] | tuple[None, None]:
    video_urls = media.get("videos", [])
    image_urls = media.get("images", [])

    async with aiohttp.ClientSession() as session:
        if video_urls:
            for url in video_urls:
                data = await download_bytes(session, url)
                if data:
                    return "video", data

        collage = await build_collage(image_urls)
        if collage:
            return "photo", collage

        if image_urls:
            for url in image_urls:
                data = await download_bytes(session, url)
                if data:
                    return "photo", data

    return None, None


def post_actions_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Опублікувати", callback_data="publish_post"),
        InlineKeyboardButton(text="❌ Скасувати", callback_data="cancel_post"),
    )
    return builder.as_markup()


def poll_actions_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Опублікувати", callback_data="publish_poll"),
        InlineKeyboardButton(text="❌ Скасувати", callback_data="cancel_poll"),
    )
    return builder.as_markup()


async def is_bot_admin(bot: Bot) -> bool:
    member = await bot.get_chat_member(chat_id=CHANNEL_ID, user_id=(await bot.me()).id)
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}


def require_admin_user(message_or_query: Message | CallbackQuery) -> bool:
    user_id = message_or_query.from_user.id
    return user_id == ADMIN_ID


@router.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    if not require_admin_user(message):
        await message.answer("Доступ заборонено.")
        return
    await state.clear()
    await message.answer("Оберіть дію:", reply_markup=MAIN_MENU)


@router.message(F.text == "3️⃣ Скасувати")
async def cancel_handler(message: Message, state: FSMContext) -> None:
    if not require_admin_user(message):
        return
    await state.clear()
    await message.answer("Скасовано. Оберіть дію:", reply_markup=MAIN_MENU)


@router.message(F.text == "1️⃣ Новий пост")
async def new_post_handler(message: Message, state: FSMContext) -> None:
    if not require_admin_user(message):
        return

    await state.clear()
    await message.answer("Шукаю свіжу новину...")

    entry = await fetch_entry_from_rss()
    if not entry:
        await message.answer("Нових новин без повторів не знайдено.", reply_markup=MAIN_MENU)
        return

    caption = build_caption(entry["title"], entry["text"], entry["game_link"])
    media_type, media_bytes = await pick_media(entry["media"])

    await state.set_state(PublishStates.waiting_post_confirm)
    await state.update_data(
        post_data={
            "link": entry["link"],
            "caption": caption,
            "media_type": media_type,
            "media_bytes": media_bytes.hex() if media_bytes else None,
        }
    )

    if media_type == "video" and media_bytes:
        video = BufferedInputFile(bytes.fromhex((await state.get_data())["post_data"]["media_bytes"]), filename="preview.mp4")
        await message.answer_video(video=video, caption=caption)
    elif media_type == "photo" and media_bytes:
        photo = BufferedInputFile(bytes.fromhex((await state.get_data())["post_data"]["media_bytes"]), filename="preview.jpg")
        await message.answer_photo(photo=photo, caption=caption)
    else:
        await message.answer(caption)

    await message.answer("Підтвердити публікацію?", reply_markup=post_actions_keyboard())


@router.callback_query(F.data == "cancel_post")
async def cancel_post_callback(query: CallbackQuery, state: FSMContext) -> None:
    if not require_admin_user(query):
        await query.answer("Недостатньо прав", show_alert=True)
        return
    await state.clear()
    await query.message.answer("Публікацію скасовано.", reply_markup=MAIN_MENU)
    await query.answer()


@router.callback_query(F.data == "publish_post")
async def publish_post_callback(query: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    if not require_admin_user(query):
        await query.answer("Недостатньо прав", show_alert=True)
        return

    if not await is_bot_admin(bot):
        await query.answer("Бот не адміністратор каналу", show_alert=True)
        return

    data = await state.get_data()
    post_data = data.get("post_data")
    if not post_data:
        await query.answer("Дані поста відсутні", show_alert=True)
        return

    media_type = post_data.get("media_type")
    media_hex = post_data.get("media_bytes")
    caption = post_data.get("caption", "")

    try:
        if media_type == "video" and media_hex:
            video = BufferedInputFile(bytes.fromhex(media_hex), filename="post.mp4")
            await bot.send_video(chat_id=CHANNEL_ID, video=video, caption=caption)
        elif media_type == "photo" and media_hex:
            photo = BufferedInputFile(bytes.fromhex(media_hex), filename="post.jpg")
            await bot.send_photo(chat_id=CHANNEL_ID, photo=photo, caption=caption)
        else:
            await bot.send_message(chat_id=CHANNEL_ID, text=caption)

        mark_published(post_data["link"])
        await state.clear()
        await query.message.answer("Пост опубліковано ✅", reply_markup=MAIN_MENU)
        await query.answer()
    except Exception as exc:
        logger.exception("Помилка публікації поста: %s", exc)
        await query.answer("Не вдалося опублікувати пост", show_alert=True)


@router.message(F.text == "2️⃣ Опитування")
async def poll_handler(message: Message, state: FSMContext) -> None:
    if not require_admin_user(message):
        return

    await state.clear()
    template = POLL_TEMPLATES[0]
    storage = load_storage()
    poll_idx = storage.get("poll_idx", 0) % len(POLL_TEMPLATES)
    template = POLL_TEMPLATES[poll_idx]
    storage["poll_idx"] = poll_idx + 1
    save_storage(storage)

    await state.set_state(PublishStates.waiting_poll_confirm)
    await state.update_data(poll_data=template)

    await message.answer("Попередній перегляд опитування:")
    await message.answer_poll(
        question=template["question"],
        options=template["options"],
        is_anonymous=False,
        allow_sending_without_reply=True,
    )
    await message.answer("Підтвердити публікацію?", reply_markup=poll_actions_keyboard())


@router.callback_query(F.data == "cancel_poll")
async def cancel_poll_callback(query: CallbackQuery, state: FSMContext) -> None:
    if not require_admin_user(query):
        await query.answer("Недостатньо прав", show_alert=True)
        return
    await state.clear()
    await query.message.answer("Опитування скасовано.", reply_markup=MAIN_MENU)
    await query.answer()


@router.callback_query(F.data == "publish_poll")
async def publish_poll_callback(query: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    if not require_admin_user(query):
        await query.answer("Недостатньо прав", show_alert=True)
        return

    if not await is_bot_admin(bot):
        await query.answer("Бот не адміністратор каналу", show_alert=True)
        return

    data = await state.get_data()
    poll_data = data.get("poll_data")
    if not poll_data:
        await query.answer("Дані опитування відсутні", show_alert=True)
        return

    try:
        await bot.send_poll(
            chat_id=CHANNEL_ID,
            question=poll_data["question"],
            options=poll_data["options"],
            is_anonymous=False,
            allow_sending_without_reply=True,
        )
        await state.clear()
        await query.message.answer("Опитування опубліковано ✅", reply_markup=MAIN_MENU)
        await query.answer()
    except Exception as exc:
        logger.exception("Помилка публікації опитування: %s", exc)
        await query.answer("Не вдалося опублікувати опитування", show_alert=True)


@router.message()
async def fallback_handler(message: Message) -> None:
    if not require_admin_user(message):
        return
    await message.answer("Оберіть дію з меню.", reply_markup=MAIN_MENU)


async def on_startup(bot: Bot) -> None:
    ensure_storage()
    if not BOT_TOKEN or not CHANNEL_ID:
        raise RuntimeError("BOT_TOKEN або CHANNEL_ID не задані")
    logger.info("Бот запущено")
    await bot.send_message(chat_id=ADMIN_ID, text="Бот запущено і готовий до роботи ✅")


async def main() -> None:
    ensure_storage()
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    @dp.errors()
    async def error_handler(event: Any) -> bool:
        logger.exception("Помилка обробки: %s", event.exception)
        return True

    await on_startup(bot)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
