import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    bot_token: str
    admin_id: int
    channel_id: str
    channel_invite_url: str
    storage_path: str = "storage.json"



def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


config = Config(
    bot_token=_require_env("BOT_TOKEN"),
    admin_id=int(_require_env("ADMIN_ID")),
    channel_id=_require_env("CHANNEL_ID"),
    channel_invite_url=os.getenv("CHANNEL_INVITE_URL", "https://t.me/hhh_play").strip() or "https://t.me/hhh_play",
)

RSS_FEEDS = [
    "https://feeds.ign.com/ign/games-all",
    "https://www.gamespot.com/feeds/mashup/",
    "https://www.pcgamer.com/rss/",
    "https://www.polygon.com/rss/index.xml",
    "https://www.eurogamer.net/rss",
    "https://kotaku.com/rss",
    "https://www.rockpapershotgun.com/feed",
]

POLL_TEMPLATES = [
    {
        "question": "🎮 Який жанр зараз у твоєму топі?",
        "options": ["RPG", "Шутери", "Стратегії", "Інді"],
    },
    {
        "question": "🔥 Що для тебе найважливіше в грі?",
        "options": ["Сюжет", "Геймплей", "Графіка", "Онлайн"],
    },
    {
        "question": "🕹️ Коли найчастіше граєш?",
        "options": ["Зранку", "Вдень", "Увечері", "Вночі"],
    },
    {
        "question": "🏆 Який формат проходження тобі ближчий?",
        "options": ["Соло", "Кооп", "PvP", "Змішаний"],
    },
]
