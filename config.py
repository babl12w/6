import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ADMIN_ID = int(os.getenv("ADMIN_ID"))

RSS_FEEDS = [
    "https://www.ign.com/rss",
    "https://www.gamespot.com/feeds/game-news/",
    "https://www.pcgamer.com/rss/",
    "https://www.eurogamer.net/feed",
    "https://kotaku.com/rss",
]

PROMO_LINK = "https://t.me/hhh_play"
STORAGE_FILE = "storage.json"
MAX_POST_LINES = 10
MIN_POST_LINES = 2
