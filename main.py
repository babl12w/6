import html
import json
import logging
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiohttp
import feedparser
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatMemberStatus, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from PIL import Image

from config import POLL_TEMPLATES, RSS_FEEDS, config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="1️⃣ Новий пост")],
        [KeyboardButton(text="2️⃣ Опитування")],
        [KeyboardButton(text="3️⃣ Скасувати")],
    ],
    resize_keyboard=True,
)

ACTION_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Опублікувати", callback_data="publish"),
            InlineKeyboardButton(text="❌ Скасувати", callback_data="cancel_publish"),
        ]
    ]
)


@dataclass
class PendingPost:
    title: str
    text: str
    source_url: str
    media_type: str
    media_path: str | None
    caption: str


pending_posts: dict[int, PendingPost] = {}


class JsonStorage:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text(json.dumps({"posted_links": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def has(self, link: str) -> bool:
        data = self._read()
        return link in set(data.get("posted_links", []))

    def add(self, link: str) -> None:
        data = self._read()
        posted = set(data.get("posted_links", []))
        posted.add(link)
        data["posted_links"] = sorted(posted)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


storage = JsonStorage(config.storage_path)


def esc(value: str) -> str:
    return html.escape(value or "", quote=False)


def fetch_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            link = entry.get("link", "").strip()
            title = entry.get("title", "").strip()
            if not link or not title:
                continue
            entries.append(
                {
                    "title": title,
                    "summary": entry.get("summary", "") or entry.get("description", ""),
                    "link": link,
                    "entry": entry,
                }
            )
    random.shuffle(entries)
    return entries


def to_ukrainian(text: str) -> str:
    clean = BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)
    if not clean:
        return ""
    translated = GoogleTranslator(source="auto", target="ukrainian").translate(clean)
    return translated.strip()


def split_lines(text: str, min_lines: int = 2, max_lines: int = 10) -> str:
    chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not chunks:
        return text
    lines = chunks[:max_lines]
    if len(lines) < min_lines and len(chunks) > len(lines):
        lines = chunks[:min(max_lines, min_lines)]
    return "\n".join(lines)


def extract_media_urls(entry: dict[str, Any], base_link: str) -> tuple[list[str], list[str]]:
    image_urls: list[str] = []
    video_urls: list[str] = []

    def push_unique(target: list[str], url: str) -> None:
        full = urljoin(base_link, url.strip())
        if full and full not in target:
            target.append(full)

    for media in entry.get("media_content", []) or []:
        url = media.get("url", "")
        mtype = (media.get("type") or "").lower()
        if "video" in mtype:
            push_unique(video_urls, url)
        elif "image" in mtype or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            push_unique(image_urls, url)

    for media in entry.get("links", []) or []:
        href = media.get("href", "")
        mtype = (media.get("type") or "").lower()
        if "video" in mtype:
            push_unique(video_urls, href)
        elif "image" in mtype:
            push_unique(image_urls, href)

    html_blocks = []
    if entry.get("summary"):
        html_blocks.append(entry["summary"])
    for part in entry.get("content", []) or []:
        if isinstance(part, dict) and part.get("value"):
            html_blocks.append(part["value"])

    for block in html_blocks:
        soup = BeautifulSoup(block, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                push_unique(image_urls, src)
        for video in soup.find_all("video"):
            src = video.get("src")
            if src:
                push_unique(video_urls, src)
            for source in video.find_all("source"):
                s = source.get("src")
                if s:
                    push_unique(video_urls, s)

    return image_urls[:10], video_urls[:5]


async def download_file(session: aiohttp.ClientSession, url: str, suffix: str) -> str | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                while True:
                    chunk = await resp.content.read(1024 * 64)
                    if not chunk:
                        break
                    tmp.write(chunk)
                return tmp.name
    except Exception:
        return None


async def download_images(session: aiohttp.ClientSession, image_urls: list[str]) -> list[str]:
    files: list[str] = []
    for url in image_urls:
        ext = ".jpg"
        low = url.lower()
        if low.endswith(".png"):
            ext = ".png"
        elif low.endswith(".webp"):
            ext = ".webp"
        path = await download_file(session, url, ext)
        if path:
            files.append(path)
        if len(files) >= 3:
            break
    return files


def create_collage(paths: list[str]) -> str:
    images = []
    for p in paths[:3]:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            continue
    if len(images) < 3:
        raise RuntimeError("not enough valid images for collage")

    width, height = 1200, 1200
    cell_w = width // 3
    collage = Image.new("RGB", (width, height), color=(20, 20, 20))

    for i, img in enumerate(images[:3]):
        ratio = max(cell_w / img.width, height / img.height)
        resized = img.resize((int(img.width * ratio), int(img.height * ratio)))
        left = (resized.width - cell_w) // 2
        top = (resized.height - height) // 2
        crop = resized.crop((left, top, left + cell_w, top + height))
        collage.paste(crop, (i * cell_w, 0))

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    out.close()
    collage.save(out.name, format="JPEG", quality=90)
    for img in images:
        img.close()
    return out.name


def cleanup(paths: list[str]) -> None:
    for p in paths:
        if p and Path(p).exists():
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def build_post(title_uk: str, body_uk: str, article_link: str) -> str:
    linked = f'<a href="{esc(article_link)}">джерело</a>'
    body = split_lines(body_uk)
    text = (
        f"<b>🎮 {esc(title_uk)}</b>\n\n"
        f"{esc(body)}\n\n"
        f"Детальніше: {linked}\n\n"
        f'🔗 <a href="{esc(config.channel_invite_url)}">Темні Ігри | Підписуйся</a>'
    )
    return text.strip()


async def generate_news_post() -> PendingPost:
    entries = await asyncio.to_thread(fetch_entries)
    if not entries:
        raise RuntimeError("Не вдалося отримати новини з RSS")

    async with aiohttp.ClientSession() as session:
        for item in entries:
            if storage.has(item["link"]):
                continue

            title_uk = await asyncio.to_thread(to_ukrainian, item["title"])
            body_source = item.get("summary") or item["title"]
            body_uk = await asyncio.to_thread(to_ukrainian, body_source)
            message_text = build_post(title_uk or item["title"], body_uk or item["title"], item["link"])

            image_urls, video_urls = extract_media_urls(item["entry"], item["link"])

            created_files: list[str] = []

            if image_urls:
                imgs = await download_images(session, image_urls)
                created_files.extend(imgs)
                if len(imgs) >= 3:
                    try:
                        collage = await asyncio.to_thread(create_collage, imgs[:3])
                        created_files.append(collage)
                        return PendingPost(
                            title=item["title"],
                            text=body_uk,
                            source_url=item["link"],
                            media_type="photo",
                            media_path=collage,
                            caption=message_text,
                        )
                    finally:
                        cleanup(imgs)
                if imgs:
                    keep = imgs[0]
                    cleanup(imgs[1:])
                    return PendingPost(
                        title=item["title"],
                        text=body_uk,
                        source_url=item["link"],
                        media_type="photo",
                        media_path=keep,
                        caption=message_text,
                    )

            for vurl in video_urls:
                video = await download_file(session, vurl, ".mp4")
                if video:
                    return PendingPost(
                        title=item["title"],
                        text=body_uk,
                        source_url=item["link"],
                        media_type="video",
                        media_path=video,
                        caption=message_text,
                    )

            return PendingPost(
                title=item["title"],
                text=body_uk,
                source_url=item["link"],
                media_type="text",
                media_path=None,
                caption=message_text,
            )

    raise RuntimeError("Усі новини вже були опубліковані")


async def send_with_caption_guard(bot: Bot, chat_id: int | str, post: PendingPost) -> None:
    caption = post.caption
    if post.media_type in {"photo", "video"} and post.media_path:
        media_file = FSInputFile(post.media_path)
        if len(caption) <= 1000:
            try:
                if post.media_type == "photo":
                    await bot.send_photo(chat_id=chat_id, photo=media_file, caption=caption)
                else:
                    await bot.send_video(chat_id=chat_id, video=media_file, caption=caption)
                return
            except TelegramBadRequest as e:
                if "caption is too long" not in str(e).lower():
                    raise
        if post.media_type == "photo":
            await bot.send_photo(chat_id=chat_id, photo=media_file)
        else:
            await bot.send_video(chat_id=chat_id, video=media_file)
        await bot.send_message(chat_id=chat_id, text=caption)
        return

    await bot.send_message(chat_id=chat_id, text=caption)


async def ensure_bot_can_post(bot: Bot) -> bool:
    me = await bot.get_me()
    member = await bot.get_chat_member(config.channel_id, me.id)
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}


async def cmd_start(message: Message) -> None:
    if message.from_user.id != config.admin_id:
        await message.answer("Доступ заборонено")
        return
    await message.answer("Меню керування:", reply_markup=MENU)


async def new_post(message: Message, bot: Bot) -> None:
    if message.from_user.id != config.admin_id:
        return
    await message.answer("Генерую пост...")
    try:
        post = await generate_news_post()
        pending_posts[message.from_user.id] = post
        await send_with_caption_guard(bot, message.chat.id, post)
        await message.answer("Публікуємо?", reply_markup=ACTION_KB)
    except Exception as e:
        logger.exception("new_post_error")
        await message.answer(f"Помилка: {e}")


async def send_poll(message: Message) -> None:
    if message.from_user.id != config.admin_id:
        return
    poll = random.choice(POLL_TEMPLATES)
    await message.answer_poll(
        question=poll["question"],
        options=poll["options"],
        is_anonymous=False,
    )


async def cancel_action(message: Message) -> None:
    if message.from_user.id != config.admin_id:
        return
    post = pending_posts.pop(message.from_user.id, None)
    if post and post.media_path:
        cleanup([post.media_path])
    await message.answer("Скасовано", reply_markup=MENU)


async def on_publish(callback, bot: Bot) -> None:
    if callback.from_user.id != config.admin_id:
        await callback.answer("Недостатньо прав", show_alert=True)
        return

    post = pending_posts.get(callback.from_user.id)
    if not post:
        await callback.answer("Немає підготовленого поста", show_alert=True)
        return

    can_post = await ensure_bot_can_post(bot)
    if not can_post:
        await callback.answer("Бот не адмін у каналі", show_alert=True)
        return

    await send_with_caption_guard(bot, config.channel_id, post)
    storage.add(post.source_url)
    pending_posts.pop(callback.from_user.id, None)
    if post.media_path:
        cleanup([post.media_path])
    await callback.message.answer("Опубліковано ✅")
    await callback.answer()


async def on_cancel_publish(callback) -> None:
    post = pending_posts.pop(callback.from_user.id, None)
    if post and post.media_path:
        cleanup([post.media_path])
    await callback.message.answer("Скасовано ❌")
    await callback.answer()


async def main() -> None:
    bot = Bot(token=config.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(new_post, F.text == "1️⃣ Новий пост")
    dp.message.register(send_poll, F.text == "2️⃣ Опитування")
    dp.message.register(cancel_action, F.text == "3️⃣ Скасувати")

    dp.callback_query.register(on_publish, F.data == "publish")
    dp.callback_query.register(on_cancel_publish, F.data == "cancel_publish")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
