# 📚 Telegram Bot Code Snippets Library

## 13. Conversation Handlers & State Machines

### Multi-Step Registration Flow
**למה זה שימושי:** ניהול שיחות מורכבות עם משתמשים דרך מספר שלבים (כמו רישום, טופס, הזמנה)

```python
from telegram.ext import ConversationHandler, CommandHandler, MessageHandler, filters

# Define states
NAME, EMAIL, CONFIRM = range(3)

async def start_registration(update, context):
    await update.message.reply_text("מה שמך?")
    return NAME

async def get_name(update, context):
    context.user_data['name'] = update.message.text
    await update.message.reply_text("מעולה! מה האימייל שלך?")
    return EMAIL

async def get_email(update, context):
    context.user_data['email'] = update.message.text
    await update.message.reply_text(
        f"שם: {context.user_data['name']}\n"
        f"אימייל: {context.user_data['email']}\n"
        "לאשר? (כן/לא)"
    )
    return CONFIRM

async def confirm(update, context):
    if update.message.text.lower() == 'כן':
        # Save to database
        await update.message.reply_text("נרשמת בהצלחה! ✅")
    return ConversationHandler.END

conv_handler = ConversationHandler(
    entry_points=[CommandHandler('register', start_registration)],
    states={
        NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_name)],
        EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_email)],
        CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, confirm)],
    },
    fallbacks=[CommandHandler('cancel', lambda u, c: ConversationHandler.END)],
    conversation_timeout=300  # 5 minutes timeout
)
```

### State Persistence with Redis
**למה זה שימושי:** שמירת מצב שיחה גם אם הבוט נופל - המשתמש ממשיך מאיפה שעצר

```python
import redis
import json
from telegram.ext import PicklePersistence, BasePersistence

class RedisPersistence(BasePersistence):
    def __init__(self, redis_url):
        super().__init__()
        self.redis = redis.from_url(redis_url)
    
    async def get_user_data(self):
        data = self.redis.get('user_data')
        return json.loads(data) if data else {}
    
    async def update_user_data(self, user_id, data):
        all_data = await self.get_user_data()
        all_data[user_id] = data
        self.redis.set('user_data', json.dumps(all_data))
    
    async def get_conversation(self, name):
        data = self.redis.get(f'conv:{name}')
        return json.loads(data) if data else {}
    
    async def update_conversation(self, name, key, new_state):
        convs = await self.get_conversation(name)
        if new_state is None:
            convs.pop(key, None)
        else:
            convs[key] = new_state
        self.redis.set(f'conv:{name}', json.dumps(convs))

# Usage
persistence = RedisPersistence('redis://localhost:6379')
app = Application.builder().token(TOKEN).persistence(persistence).build()
```

### Nested Conversation with Sub-States
**למה זה שימושי:** שיחות מורכבות עם תפריטים ותת-תפריטים (כמו הזמנת פיצה עם תוספות)

```python
CHOOSE_PIZZA, CHOOSE_SIZE, CHOOSE_TOPPINGS, CONFIRM_ORDER = range(4)
TOPPING_ADD, TOPPING_DONE = range(4, 6)

async def pizza_menu(update, context):
    keyboard = [['מרגריטה', 'פפרוני'], ['ירקות', 'חזרה']]
    await update.message.reply_text(
        "בחר פיצה:", 
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    )
    return CHOOSE_PIZZA

async def choose_toppings(update, context):
    context.user_data['pizza'] = update.message.text
    context.user_data['toppings'] = []
    await update.message.reply_text(
        "הוסף תוספות (או שלח 'סיימתי'):"
    )
    return TOPPING_ADD

async def add_topping(update, context):
    if update.message.text == 'סיימתי':
        return await show_order_summary(update, context)
    
    context.user_data['toppings'].append(update.message.text)
    await update.message.reply_text(f"✅ נוסף {update.message.text}. עוד תוספות?")
    return TOPPING_ADD

nested_handler = ConversationHandler(
    entry_points=[CommandHandler('order', pizza_menu)],
    states={
        CHOOSE_PIZZA: [MessageHandler(filters.TEXT, choose_toppings)],
        TOPPING_ADD: [MessageHandler(filters.TEXT, add_topping)],
    },
    fallbacks=[CommandHandler('cancel', cancel_order)]
)
```

---

## 14. Telegram Media Handling

### Image Processing & Resize Before Send
**למה זה שימושי:** הקטנת תמונות כדי לא לחרוג מגבולות טלגרם ולשפר מהירות העלאה

```python
from PIL import Image
from io import BytesIO

async def resize_and_send_image(update, context, image_path, max_size=(1280, 1280)):
    """
    Resize image if needed and send to user
    """
    img = Image.open(image_path)
    
    # Resize if too large
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to bytes
    bio = BytesIO()
    img.save(bio, format='JPEG', quality=85, optimize=True)
    bio.seek(0)
    
    await update.message.reply_photo(
        photo=bio,
        caption="תמונה מעובדת ומוקטנת"
    )

# Handle received images
async def process_received_image(update, context):
    photo = update.message.photo[-1]  # Largest size
    file = await photo.get_file()
    
    bio = BytesIO()
    await file.download_to_memory(bio)
    bio.seek(0)
    
    img = Image.open(bio)
    # Process image...
    width, height = img.size
    await update.message.reply_text(f"התמונה שלך: {width}x{height}px")
```

### Video Thumbnail Generation
**למה זה שימושי:** יצירת תמונות תצוגה מקדימה לסרטונים באופן אוטומטי

```python
import cv2
from pathlib import Path

async def generate_video_thumbnail(video_path, timestamp=1.0):
    """
    Extract frame from video at given timestamp
    """
    cap = cv2.VideoCapture(str(video_path))
    
    # Set position to timestamp (in seconds)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * timestamp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Save as thumbnail
        thumb_path = Path(video_path).with_suffix('.jpg')
        cv2.imwrite(str(thumb_path), frame)
        return thumb_path
    return None

async def send_video_with_thumb(update, context, video_path):
    thumb = await generate_video_thumbnail(video_path)
    
    with open(video_path, 'rb') as video:
        await update.message.reply_video(
            video=video,
            thumbnail=open(thumb, 'rb') if thumb else None,
            caption="וידאו עם תמונה ממוזערת"
        )
```

### Smart Document Type Detection
**למה זה שימושי:** זיהוי אוטומטי של סוג קובץ וטיפול מתאים לכל סוג

```python
import magic
from pathlib import Path

class DocumentHandler:
    HANDLERS = {
        'application/pdf': 'handle_pdf',
        'image/': 'handle_image',
        'video/': 'handle_video',
        'text/': 'handle_text',
    }
    
    async def handle_document(self, update, context):
        doc = update.message.document
        file = await doc.get_file()
        
        # Download file
        file_path = f"/tmp/{doc.file_name}"
        await file.download_to_drive(file_path)
        
        # Detect MIME type
        mime = magic.from_file(file_path, mime=True)
        
        # Route to appropriate handler
        for pattern, handler_name in self.HANDLERS.items():
            if mime.startswith(pattern):
                handler = getattr(self, handler_name)
                await handler(update, context, file_path)
                return
        
        await update.message.reply_text(f"סוג קובץ לא נתמך: {mime}")
    
    async def handle_pdf(self, update, context, path):
        await update.message.reply_text("📄 מעבד PDF...")
        # Process PDF
    
    async def handle_image(self, update, context, path):
        await update.message.reply_text("🖼 מעבד תמונה...")
        # Process image
```

### Media Compression Before Upload
**למה זה שימושי:** הקטנת גודל קבצים לפני העלאה - חסכון בזמן ובנתונים

```python
from moviepy.editor import VideoFileClip
import subprocess

async def compress_video(input_path, output_path, target_size_mb=10):
    """
    Compress video to target file size using ffmpeg
    """
    # Get original duration
    clip = VideoFileClip(input_path)
    duration = clip.duration
    clip.close()
    
    # Calculate target bitrate
    target_bitrate = int((target_size_mb * 8192) / duration)
    
    # Compress using ffmpeg
    cmd = [
        'ffmpeg', '-i', input_path,
        '-b:v', f'{target_bitrate}k',
        '-maxrate', f'{target_bitrate}k',
        '-bufsize', f'{target_bitrate * 2}k',
        '-vcodec', 'libx264',
        '-preset', 'medium',
        '-y', output_path
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

async def send_compressed_video(update, context, video_path):
    compressed = await compress_video(video_path, '/tmp/compressed.mp4')
    
    with open(compressed, 'rb') as video:
        await update.message.reply_video(
            video=video,
            caption="וידאו דחוס"
        )
```

---

## 15. Internationalization (i18n)

### Multi-Language Support System
**למה זה שימושי:** תמיכה בריבוי שפות - כל משתמש רואה את הבוט בשפה שלו

```python
import json
from pathlib import Path

class I18n:
    def __init__(self, locales_dir='locales'):
        self.locales_dir = Path(locales_dir)
        self.translations = {}
        self._load_translations()
    
    def _load_translations(self):
        for locale_file in self.locales_dir.glob('*.json'):
            lang = locale_file.stem
            with open(locale_file, 'r', encoding='utf-8') as f:
                self.translations[lang] = json.load(f)
    
    def t(self, key, lang='en', **kwargs):
        """Get translation for key"""
        text = self.translations.get(lang, {}).get(key, key)
        return text.format(**kwargs) if kwargs else text
    
    def get_user_lang(self, user):
        """Detect user language from Telegram"""
        return user.language_code or 'en'

# Initialize
i18n = I18n('locales')

async def start(update, context):
    lang = i18n.get_user_lang(update.effective_user)
    greeting = i18n.t('greeting', lang, name=update.effective_user.first_name)
    await update.message.reply_text(greeting)

# locales/he.json
# {
#   "greeting": "שלום {name}! ברוך הבא לבוט",
#   "help": "אני יכול לעזור לך ב...",
#   "error": "אופס! משהו השתבש"
# }

# locales/en.json
# {
#   "greeting": "Hello {name}! Welcome to the bot",
#   "help": "I can help you with...",
#   "error": "Oops! Something went wrong"
# }
```

### Language Switcher with Inline Keyboard
**למה זה שימושי:** אפשרות למשתמש לבחור שפה בקלות

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

LANGUAGES = {
    'en': '🇬🇧 English',
    'he': '🇮🇱 עברית',
    'es': '🇪🇸 Español',
    'ru': '🇷🇺 Русский'
}

async def change_language(update, context):
    keyboard = [
        [InlineKeyboardButton(name, callback_data=f'lang:{code}')]
        for code, name in LANGUAGES.items()
    ]
    await update.message.reply_text(
        "Choose language / בחר שפה:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def language_callback(update, context):
    query = update.callback_query
    lang = query.data.split(':')[1]
    
    # Save to database
    await save_user_language(query.from_user.id, lang)
    
    await query.answer()
    await query.edit_message_text(
        i18n.t('language_changed', lang)
    )
```

### RTL/LTR Text Handling
**למה זה שימושי:** תצוגה נכונה של טקסט בשפות RTL כמו עברית וערבית

```python
from telegram.constants import ParseMode

def format_rtl_text(text, lang='he'):
    """
    Add RTL marks for proper text display
    """
    RTL_LANGS = ['he', 'ar', 'fa']
    
    if lang in RTL_LANGS:
        # Add Right-to-Left Mark
        return f"\u200F{text}\u200F"
    return text

async def send_mixed_text(update, context, lang):
    """
    Send message with proper direction marks
    """
    name = "John Doe"  # Latin text
    message = i18n.t('welcome_msg', lang, name=name)
    
    # Format with RTL if needed
    formatted = format_rtl_text(message, lang)
    
    await update.message.reply_text(
        formatted,
        parse_mode=ParseMode.HTML
    )

# Advanced: Mixed content handling
def wrap_ltr_in_rtl(text):
    """
    Wrap English words in RTL context
    """
    import re
    # Find Latin text and wrap with LTR marks
    pattern = r'[a-zA-Z0-9@._-]+'
    return re.sub(pattern, r'\u200E\g<0>\u200E', text)

# Example: "שלום John, יש לך 5 הודעות חדשות"
# Becomes: "שלום ‎John‎, יש לך ‎5‎ הודעות חדשות"
```

---

## 16. Command Permissions & Access Control

### Role-Based Access Control Decorator
**למה זה שימושי:** הגבלת גישה לפקודות לפי תפקיד משתמש (אדמין, VIP, משתמש רגיל)

```python
from functools import wraps
from enum import Enum

class Role(Enum):
    USER = 1
    VIP = 2
    ADMIN = 3

# Mock database
USER_ROLES = {
    12345: Role.ADMIN,
    67890: Role.VIP,
}

def require_role(min_role: Role):
    def decorator(func):
        @wraps(func)
        async def wrapper(update, context):
            user_id = update.effective_user.id
            user_role = USER_ROLES.get(user_id, Role.USER)
            
            if user_role.value < min_role.value:
                await update.message.reply_text(
                    "⛔️ אין לך הרשאה לפקודה זו"
                )
                return
            
            return await func(update, context)
        return wrapper
    return decorator

# Usage
@require_role(Role.ADMIN)
async def admin_panel(update, context):
    await update.message.reply_text("🔐 פאנל אדמין")

@require_role(Role.VIP)
async def vip_feature(update, context):
    await update.message.reply_text("⭐️ פיצ'ר VIP")
```

### User Whitelist/Blacklist System
**למה זה שימושי:** חסימה או אישור של משתמשים ספציפיים

```python
import redis

class AccessControl:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def add_to_whitelist(self, user_id):
        self.redis.sadd('whitelist', user_id)
    
    def add_to_blacklist(self, user_id):
        self.redis.sadd('blacklist', user_id)
    
    def is_allowed(self, user_id):
        # Check blacklist first
        if self.redis.sismember('blacklist', user_id):
            return False
        
        # If whitelist exists and not empty, check membership
        if self.redis.scard('whitelist') > 0:
            return self.redis.sismember('whitelist', user_id)
        
        # Default: allow
        return True

# Initialize
acl = AccessControl(redis.Redis())

def require_access(func):
    @wraps(func)
    async def wrapper(update, context):
        user_id = update.effective_user.id
        
        if not acl.is_allowed(user_id):
            await update.message.reply_text("🚫 אין לך גישה לבוט זה")
            return
        
        return await func(update, context)
    return wrapper

# Admin commands to manage access
@require_role(Role.ADMIN)
async def whitelist_user(update, context):
    user_id = int(context.args[0])
    acl.add_to_whitelist(user_id)
    await update.message.reply_text(f"✅ משתמש {user_id} נוסף לרשימה הלבנה")

@require_role(Role.ADMIN)
async def blacklist_user(update, context):
    user_id = int(context.args[0])
    acl.add_to_blacklist(user_id)
    await update.message.reply_text(f"🚫 משתמש {user_id} נחסם")
```

### Permission Caching for Performance
**למה זה שימושי:** שמירת הרשאות במטמון כדי לא לבדוק בDB בכל בקשה

```python
from functools import lru_cache
from datetime import datetime, timedelta

class PermissionCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, user_id):
        if user_id in self.cache:
            cached_data, timestamp = self.cache[user_id]
            if datetime.now() - timestamp < self.ttl:
                return cached_data
            del self.cache[user_id]
        return None
    
    def set(self, user_id, permissions):
        self.cache[user_id] = (permissions, datetime.now())
    
    def invalidate(self, user_id):
        self.cache.pop(user_id, None)

perm_cache = PermissionCache(ttl_seconds=600)  # 10 minutes

async def check_permission(user_id, permission):
    # Try cache first
    cached = perm_cache.get(user_id)
    if cached:
        return permission in cached
    
    # Fetch from database
    user_perms = await db.get_user_permissions(user_id)
    
    # Cache for next time
    perm_cache.set(user_id, user_perms)
    
    return permission in user_perms

def require_permission(permission):
    def decorator(func):
        @wraps(func)
        async def wrapper(update, context):
            user_id = update.effective_user.id
            
            if not await check_permission(user_id, permission):
                await update.message.reply_text("⛔️ אין לך הרשאה")
                return
            
            return await func(update, context)
        return wrapper
    return decorator

# Usage
@require_permission('delete_posts')
async def delete_post(update, context):
    await update.message.reply_text("🗑 פוסט נמחק")
```

---

## 17. Graceful Shutdown & Cleanup

### Complete Shutdown Handler
**למה זה שימושי:** סגירה נקייה של הבוט - סיום תהליכים, סגירת חיבורים, שמירת מצב

```python
import signal
import asyncio
from telegram.ext import Application

class BotShutdown:
    def __init__(self, app: Application):
        self.app = app
        self.is_shutting_down = False
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        if not self.is_shutting_down:
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            self.is_shutting_down = True
            asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Perform graceful shutdown"""
        # 1. Stop accepting new updates
        await self.app.updater.stop()
        
        # 2. Wait for active handlers to complete
        print("⏳ Waiting for handlers to complete...")
        await asyncio.sleep(2)
        
        # 3. Close database connections
        print("💾 Closing database connections...")
        await db.close()
        
        # 4. Close Redis connections
        print("🔴 Closing Redis...")
        await redis_client.close()
        
        # 5. Save state
        print("💾 Saving state...")
        await save_bot_state()
        
        # 6. Stop the application
        await self.app.stop()
        await self.app.shutdown()
        
        print("✅ Shutdown complete")

# Usage
app = Application.builder().token(TOKEN).build()
shutdown_handler = BotShutdown(app)

app.run_polling()
```

### In-Flight Request Tracking
**למה זה שימושי:** מעקב אחר בקשות פעילות כדי לא לנתק אותן באמצע

```python
import asyncio
from contextlib import asynccontextmanager

class RequestTracker:
    def __init__(self):
        self.active_requests = set()
        self.shutdown_event = asyncio.Event()
    
    @asynccontextmanager
    async def track_request(self, request_id):
        """Context manager to track active requests"""
        self.active_requests.add(request_id)
        try:
            yield
        finally:
            self.active_requests.remove(request_id)
            
            # If shutting down and no more requests, signal completion
            if self.shutdown_event.is_set() and not self.active_requests:
                print("✅ All requests completed")
    
    async def wait_for_completion(self, timeout=30):
        """Wait for all requests to complete"""
        self.shutdown_event.set()
        
        if not self.active_requests:
            return True
        
        print(f"⏳ Waiting for {len(self.active_requests)} active requests...")
        
        try:
            await asyncio.wait_for(
                self._wait_empty(), 
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            print(f"⚠️  Timeout: {len(self.active_requests)} requests still active")
            return False
    
    async def _wait_empty(self):
        while self.active_requests:
            await asyncio.sleep(0.1)

tracker = RequestTracker()

async def long_running_handler(update, context):
    request_id = f"{update.effective_user.id}_{update.update_id}"
    
    async with tracker.track_request(request_id):
        # Long operation
        await update.message.reply_text("מתחיל עיבוד...")
        await asyncio.sleep(10)  # Simulate work
        await update.message.reply_text("הסתיים!")

# In shutdown
async def graceful_shutdown():
    print("🛑 Starting shutdown...")
    completed = await tracker.wait_for_completion(timeout=30)
    
    if not completed:
        print("⚠️  Force closing remaining requests")
```

### Shutdown Hooks System
**למה זה שימושי:** רישום פונקציות ניקוי שירוצו בסגירה

```python
class ShutdownHooks:
    def __init__(self):
        self.hooks = []
    
    def register(self, func, *args, **kwargs):
        """Register a cleanup function"""
        self.hooks.append((func, args, kwargs))
    
    async def run_all(self):
        """Execute all hooks"""
        print(f"🧹 Running {len(self.hooks)} cleanup hooks...")
        
        for func, args, kwargs in reversed(self.hooks):  # LIFO order
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
                print(f"✅ {func.__name__}")
            except Exception as e:
                print(f"❌ {func.__name__}: {e}")

hooks = ShutdownHooks()

# Register cleanup functions
hooks.register(db.close)
hooks.register(redis_client.close)
hooks.register(close_file_handles)
hooks.register(save_metrics)

# On shutdown
async def shutdown():
    await hooks.run_all()
```

---

## 18. Batch Processing & Bulk Operations

### Batch Database Inserts
**למה זה שימושי:** הכנסת אלפי רשומות ל-DB בבת אחת במקום אחת-אחת (פי 100 מהר יותר!)

```python
async def batch_insert_users(users, batch_size=1000):
    """
    Insert users in batches for better performance
    """
    from motor.motor_asyncio import AsyncIOMotorClient
    
    db = AsyncIOMotorClient()['mybot']
    collection = db['users']
    
    total = len(users)
    inserted = 0
    
    for i in range(0, total, batch_size):
        batch = users[i:i + batch_size]
        
        try:
            result = await collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
            print(f"✅ Inserted {inserted}/{total}")
        except Exception as e:
            print(f"❌ Batch {i} failed: {e}")
    
    return inserted

# Usage
users = [
    {'user_id': i, 'name': f'User{i}', 'joined': datetime.now()}
    for i in range(10000)
]

await batch_insert_users(users)
```

### Bulk Message Sending with Rate Limiting
**למה זה שימושי:** שליחת הודעות לאלפי משתמשים בלי לחרוג ממגבלות טלגרם

```python
import asyncio
from telegram.error import RetryAfter, TelegramError

async def bulk_send_messages(bot, user_ids, text, delay=0.05):
    """
    Send message to multiple users with rate limiting
    30 messages/second for different users
    """
    results = {'success': 0, 'failed': 0, 'blocked': 0}
    
    for user_id in user_ids:
        try:
            await bot.send_message(chat_id=user_id, text=text)
            results['success'] += 1
            await asyncio.sleep(delay)  # Rate limit
            
        except RetryAfter as e:
            print(f"⏸ Rate limited, waiting {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            # Retry
            await bot.send_message(chat_id=user_id, text=text)
            
        except TelegramError as e:
            if 'blocked' in str(e).lower():
                results['blocked'] += 1
            else:
                results['failed'] += 1
                print(f"❌ Failed {user_id}: {e}")
    
    return results

# Usage with progress bar
async def send_broadcast(bot, message):
    users = await db.get_all_user_ids()
    print(f"📢 Sending to {len(users)} users...")
    
    results = await bulk_send_messages(bot, users, message)
    
    print(f"✅ Success: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"🚫 Blocked: {results['blocked']}")
```

### Chunked Processing with Progress
**למה זה שימושי:** עיבוד כמויות גדולות של נתונים בחלקים עם מעקב התקדמות

```python
async def process_in_chunks(items, process_func, chunk_size=100, progress_callback=None):
    """
    Process large dataset in chunks with progress tracking
    """
    total = len(items)
    processed = 0
    results = []
    
    for i in range(0, total, chunk_size):
        chunk = items[i:i + chunk_size]
        
        # Process chunk
        chunk_results = await asyncio.gather(*[
            process_func(item) for item in chunk
        ])
        
        results.extend(chunk_results)
        processed += len(chunk)
        
        # Report progress
        progress = (processed / total) * 100
        if progress_callback:
            await progress_callback(processed, total, progress)
        
        print(f"⏳ {processed}/{total} ({progress:.1f}%)")
    
    return results

# Usage example
async def process_image(image_data):
    # Simulate processing
    await asyncio.sleep(0.1)
    return f"processed_{image_data['id']}"

async def progress_update(current, total, percent):
    # Update bot message with progress
    await bot.edit_message_text(
        chat_id=admin_chat,
        message_id=progress_msg_id,
        text=f"⏳ מעבד: {current}/{total} ({percent:.1f}%)"
    )

images = [{'id': i, 'url': f'img{i}.jpg'} for i in range(1000)]
results = await process_in_chunks(
    images, 
    process_image, 
    chunk_size=50,
    progress_callback=progress_update
)
```

### Batch Export to CSV
**למה זה שימושי:** ייצוא כמויות גדולות של נתונים ל-CSV ביעילות

```python
import csv
from io import StringIO

async def export_users_to_csv(batch_size=1000):
    """
    Export all users to CSV in batches (memory efficient)
    """
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['User ID', 'Name', 'Joined', 'Active'])
    
    skip = 0
    while True:
        # Fetch batch
        users = await db.users.find().skip(skip).limit(batch_size).to_list()
        
        if not users:
            break
        
        # Write batch
        for user in users:
            writer.writerow([
                user['user_id'],
                user.get('name', 'N/A'),
                user.get('joined_at', ''),
                user.get('is_active', False)
            ])
        
        skip += batch_size
        print(f"📊 Exported {skip} users...")
    
    # Get CSV content
    output.seek(0)
    return output.getvalue()

# Send to admin
async def send_user_export(update, context):
    await update.message.reply_text("⏳ מייצא נתונים...")
    
    csv_data = await export_users_to_csv()
    
    from io import BytesIO
    file = BytesIO(csv_data.encode('utf-8'))
    file.name = 'users_export.csv'
    
    await update.message.reply_document(
        document=file,
        filename='users_export.csv',
        caption=f"📊 ייצוא משתמשים"
    )
```

---

## 19. Telegram Channel/Group Management

### Auto-Moderation System
**למה זה שימושי:** ניהול אוטומטי של קבוצות - חסימת ספאם, קישורים, שפה גסה

```python
import re
from telegram import ChatPermissions

class AutoModerator:
    SPAM_PATTERNS = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'@\w+',  # Mentions
        r'(.)\1{4,}',  # Repeated characters
    ]
    
    BAD_WORDS = ['spam', 'קזינו', 'הימורים']  # Add your words
    
    async def check_message(self, update, context):
        message = update.message
        text = message.text or message.caption or ''
        
        # Check for spam patterns
        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                await self.handle_violation(message, 'spam_link')
                return True
        
        # Check for bad words
        text_lower = text.lower()
        if any(word in text_lower for word in self.BAD_WORDS):
            await self.handle_violation(message, 'bad_language')
            return True
        
        # Check flood (too many messages)
        if await self.is_flooding(message.from_user.id):
            await self.handle_violation(message, 'flooding')
            return True
        
        return False
    
    async def handle_violation(self, message, reason):
        # Delete message
        await message.delete()
        
        # Warn user
        user_id = message.from_user.id
        warnings = await db.increment_warnings(user_id)
        
        if warnings >= 3:
            # Ban user
            await message.chat.ban_member(user_id)
            await message.chat.send_message(
                f"🚫 {message.from_user.mention_html()} נחסם לצמיתות (3 אזהרות)"
            )
        else:
            # Restrict for 1 hour
            await message.chat.restrict_member(
                user_id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=datetime.now() + timedelta(hours=1)
            )
            await message.chat.send_message(
                f"⚠️ {message.from_user.mention_html()} הושתק לשעה ({reason})"
            )
    
    async def is_flooding(self, user_id):
        # Check Redis for message count in last minute
        key = f'flood:{user_id}'
        count = await redis.incr(key)
        
        if count == 1:
            await redis.expire(key, 60)
        
        return count > 10  # More than 10 messages/minute

moderator = AutoModerator()

async def group_message_handler(update, context):
    if await moderator.check_message(update, context):
        return  # Message was moderated
    
    # Process normal message
```

### Member Management Utilities
**למה זה שימושי:** כלים נוחים לניהול חברי קבוצה

```python
async def get_group_stats(chat_id):
    """Get comprehensive group statistics"""
    stats = {
        'total_members': 0,
        'admins': [],
        'bots': 0,
        'restricted': 0
    }
    
    # Get member count
    stats['total_members'] = await bot.get_chat_member_count(chat_id)
    
    # Get administrators
    admins = await bot.get_chat_administrators(chat_id)
    stats['admins'] = [
        {
            'id': admin.user.id,
            'name': admin.user.full_name,
            'status': admin.status
        }
        for admin in admins
    ]
    
    stats['bots'] = sum(1 for admin in admins if admin.user.is_bot)
    
    return stats

async def bulk_ban_users(chat_id, user_ids, reason="הפרת כללים"):
    """Ban multiple users at once"""
    results = {'success': 0, 'failed': 0}
    
    for user_id in user_ids:
        try:
            await bot.ban_chat_member(chat_id, user_id)
            results['success'] += 1
            
            # Log to admin channel
            await bot.send_message(
                admin_channel_id,
                f"🚫 User {user_id} banned from {chat_id}\nReason: {reason}"
            )
        except Exception as e:
            results['failed'] += 1
            print(f"Failed to ban {user_id}: {e}")
    
    return results

async def cleanup_inactive_members(chat_id, days=30):
    """Remove members who haven't sent a message in X days"""
    cutoff = datetime.now() - timedelta(days=days)
    inactive = await db.find({
        'chat_id': chat_id,
        'last_message': {'$lt': cutoff}
    })
    
    removed = 0
    for member in inactive:
        try:
            await bot.ban_chat_member(chat_id, member['user_id'])
            await bot.unban_chat_member(chat_id, member['user_id'])  # Soft kick
            removed += 1
        except:
            pass
    
    return removed
```

### Channel Posting Helpers
**למה זה שימושי:** כלים לפרסום אוטומטי בערוצים

```python
from telegram.constants import ParseMode

class ChannelPoster:
    def __init__(self, channel_id):
        self.channel_id = channel_id
    
    async def post_with_preview(self, title, text, image_url=None, link=None):
        """Post with link preview"""
        message = f"<b>{title}</b>\n\n{text}"
        
        if link:
            message += f"\n\n🔗 <a href='{link}'>קרא עוד</a>"
        
        if image_url:
            await bot.send_photo(
                chat_id=self.channel_id,
                photo=image_url,
                caption=message,
                parse_mode=ParseMode.HTML
            )
        else:
            await bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=False
            )
    
    async def schedule_post(self, content, scheduled_time):
        """Schedule a post for later"""
        await db.scheduled_posts.insert_one({
            'channel_id': self.channel_id,
            'content': content,
            'scheduled_for': scheduled_time,
            'status': 'pending'
        })
    
    async def post_poll(self, question, options, is_anonymous=True):
        """Post a poll"""
        await bot.send_poll(
            chat_id=self.channel_id,
            question=question,
            options=options,
            is_anonymous=is_anonymous
        )

# Background job to send scheduled posts
async def send_scheduled_posts(context):
    now = datetime.now()
    posts = await db.scheduled_posts.find({
        'scheduled_for': {'$lte': now},
        'status': 'pending'
    })
    
    for post in posts:
        try:
            await bot.send_message(
                chat_id=post['channel_id'],
                text=post['content']
            )
            
            await db.scheduled_posts.update_one(
                {'_id': post['_id']},
                {'$set': {'status': 'sent', 'sent_at': now}}
            )
        except Exception as e:
            print(f"Failed to send scheduled post: {e}")
```

---

## 20. Data Export/Import

### CSV Export with Streaming
**למה זה שימושי:** ייצוא נתונים גדולים ללא טעינת הכל לזיכרון

```python
import csv
from io import StringIO, BytesIO

async def stream_export_to_csv(query_filter=None):
    """
    Stream large dataset to CSV without loading all into memory
    """
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Username', 'Email', 'Created', 'Active'])
    
    # Stream from database
    async for user in db.users.find(query_filter or {}):
        writer.writerow([
            user['_id'],
            user.get('username', ''),
            user.get('email', ''),
            user.get('created_at', ''),
            user.get('is_active', False)
        ])
    
    # Convert to bytes for sending
    output.seek(0)
    bytes_output = BytesIO(output.getvalue().encode('utf-8-sig'))  # BOM for Excel
    bytes_output.name = 'export.csv'
    
    return bytes_output

# Command to export
async def export_data(update, context):
    await update.message.reply_text("⏳ מייצא נתונים...")
    
    csv_file = await stream_export_to_csv()
    
    await update.message.reply_document(
        document=csv_file,
        filename=f'export_{datetime.now():%Y%m%d}.csv',
        caption="✅ הנתונים שלך מוכנים!"
    )
```

### JSON Import with Validation
**למה זה שימושי:** ייבוא מאובטח של נתונים עם בדיקות תקינות

```python
import json
from pydantic import BaseModel, ValidationError
from typing import List

class UserImport(BaseModel):
    user_id: int
    username: str
    email: str | None = None
    role: str = 'user'

async def import_from_json(file_path, validate=True):
    """
    Import data from JSON with optional validation
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {
        'imported': 0,
        'failed': 0,
        'errors': []
    }
    
    for item in data:
        try:
            # Validate with Pydantic
            if validate:
                user = UserImport(**item)
                item = user.dict()
            
            # Insert to database
            await db.users.insert_one(item)
            results['imported'] += 1
            
        except ValidationError as e:
            results['failed'] += 1
            results['errors'].append({
                'item': item,
                'error': str(e)
            })
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'item': item,
                'error': str(e)
            })
    
    return results

# Handler for file upload
async def handle_import_file(update, context):
    doc = update.message.document
    
    if not doc.file_name.endswith('.json'):
        await update.message.reply_text("❌ רק קבצי JSON נתמכים")
        return
    
    # Download file
    file = await doc.get_file()
    file_path = f"/tmp/{doc.file_name}"
    await file.download_to_drive(file_path)
    
    # Import
    results = await import_from_json(file_path, validate=True)
    
    await update.message.reply_text(
        f"✅ יובאו: {results['imported']}\n"
        f"❌ נכשלו: {results['failed']}"
    )
```

### Backup/Restore Complete Database
**למה זה שימושי:** גיבוי ושחזור מלא של DB בפקודה אחת

```python
import gzip
import pickle
from datetime import datetime

async def create_backup():
    """
    Create compressed backup of entire database
    """
    backup_data = {}
    
    # Export all collections
    for collection_name in await db.list_collection_names():
        collection = db[collection_name]
        docs = await collection.find().to_list(length=None)
        backup_data[collection_name] = docs
    
    # Add metadata
    backup_data['_metadata'] = {
        'created_at': datetime.now().isoformat(),
        'version': '1.0',
        'collections': list(backup_data.keys())
    }
    
    # Compress and save
    filename = f"backup_{datetime.now():%Y%m%d_%H%M%S}.gz"
    
    with gzip.open(filename, 'wb') as f:
        pickle.dump(backup_data, f)
    
    return filename

async def restore_backup(backup_file):
    """
    Restore database from backup file
    """
    # Load backup
    with gzip.open(backup_file, 'rb') as f:
        backup_data = pickle.load(f)
    
    # Verify metadata
    metadata = backup_data.pop('_metadata', {})
    print(f"Restoring backup from {metadata.get('created_at')}")
    
    # Restore each collection
    for collection_name, docs in backup_data.items():
        collection = db[collection_name]
        
        # Clear existing data (optional)
        await collection.delete_many({})
        
        # Insert backup data
        if docs:
            await collection.insert_many(docs)
        
        print(f"✅ Restored {len(docs)} docs to {collection_name}")

# Admin commands
async def backup_command(update, context):
    await update.message.reply_text("⏳ יוצר גיבוי...")
    
    filename = await create_backup()
    
    with open(filename, 'rb') as f:
        await update.message.reply_document(
            document=f,
            filename=filename,
            caption="💾 גיבוי מלא של הנתונים"
        )

async def restore_command(update, context):
    """Usage: /restore <reply to backup file>"""
    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        await update.message.reply_text("❌ השב לקובץ גיבוי")
        return
    
    doc = update.message.reply_to_message.document
    file = await doc.get_file()
    file_path = f"/tmp/{doc.file_name}"
    await file.download_to_drive(file_path)
    
    await update.message.reply_text("⏳ משחזר נתונים...")
    
    await restore_backup(file_path)
    
    await update.message.reply_text("✅ שוחזר בהצלחה!")
```

### Data Transformation Pipeline
**למה זה שימושי:** המרה והעברת נתונים בין מבנים שונים

```python
from typing import Callable, List

class DataTransformer:
    def __init__(self):
        self.transforms: List[Callable] = []
    
    def add_transform(self, func: Callable):
        """Add transformation function to pipeline"""
        self.transforms.append(func)
        return self
    
    async def process(self, data):
        """Run all transformations on data"""
        result = data
        
        for transform in self.transforms:
            if asyncio.iscoroutinefunction(transform):
                result = await transform(result)
            else:
                result = transform(result)
        
        return result
    
    async def process_batch(self, data_list):
        """Process multiple items"""
        return [await self.process(item) for item in data_list]

# Example transformations
def normalize_phone(user):
    """Normalize phone number format"""
    if 'phone' in user:
        phone = user['phone'].replace('-', '').replace(' ', '')
        user['phone'] = phone
    return user

def add_full_name(user):
    """Combine first and last name"""
    user['full_name'] = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
    return user

async def enrich_with_location(user):
    """Add location data from external API"""
    if 'city' in user:
        location = await get_city_coordinates(user['city'])
        user['coordinates'] = location
    return user

# Build pipeline
transformer = DataTransformer()
transformer.add_transform(normalize_phone)
transformer.add_transform(add_full_name)
transformer.add_transform(enrich_with_location)

# Use it
users = await db.old_users.find().to_list()
transformed = await transformer.process_batch(users)
await db.new_users.insert_many(transformed)
```

---

## 21. Performance Monitoring

### Query Performance Tracker
**למה זה שימושי:** זיהוי שאילתות איטיות ל-DB - אופטימיזציה יעילה

```python
import time
from functools import wraps

class QueryMonitor:
    def __init__(self, slow_threshold=1.0):
        self.slow_threshold = slow_threshold
        self.slow_queries = []
    
    def track_query(self, collection_name):
        """Decorator to track query performance"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                if duration > self.slow_threshold:
                    self.slow_queries.append({
                        'collection': collection_name,
                        'function': func.__name__,
                        'duration': duration,
                        'timestamp': datetime.now()
                    })
                    print(f"⚠️  Slow query: {func.__name__} took {duration:.2f}s")
                
                return result
            return wrapper
        return decorator
    
    def get_slow_queries(self):
        """Get list of slow queries"""
        return sorted(self.slow_queries, key=lambda x: x['duration'], reverse=True)

monitor = QueryMonitor(slow_threshold=0.5)

# Usage
@monitor.track_query('users')
async def find_user_by_email(email):
    return await db.users.find_one({'email': email})

@monitor.track_query('posts')
async def get_user_posts(user_id):
    return await db.posts.find({'user_id': user_id}).to_list()

# Admin command to view slow queries
async def show_slow_queries(update, context):
    slow = monitor.get_slow_queries()[:10]
    
    if not slow:
        await update.message.reply_text("✅ אין שאילתות איטיות")
        return
    
    text = "🐌 השאילתות האיטיות:\n\n"
    for q in slow:
        text += f"• {q['collection']}.{q['function']}: {q['duration']:.2f}s\n"
    
    await update.message.reply_text(text)
```

### Memory Usage Monitor
**למה זה שימושי:** מעקב אחר צריכת זיכרון - מניעת קריסות

```python
import psutil
import os

class MemoryMonitor:
    def __init__(self, alert_threshold_mb=500):
        self.process = psutil.Process(os.getpid())
        self.alert_threshold = alert_threshold_mb * 1024 * 1024  # Convert to bytes
        self.baseline = self.get_memory_usage()
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_stats(self):
        """Get detailed memory statistics"""
        mem = self.process.memory_info()
        return {
            'rss_mb': mem.rss / 1024 / 1024,
            'vms_mb': mem.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'increase_mb': (mem.rss - self.baseline * 1024 * 1024) / 1024 / 1024
        }
    
    async def check_and_alert(self, bot, admin_chat_id):
        """Check memory and alert if threshold exceeded"""
        current = self.get_memory_usage()
        
        if current > self.alert_threshold / 1024 / 1024:
            stats = self.get_memory_stats()
            await bot.send_message(
                chat_id=admin_chat_id,
                text=f"⚠️ זיכרון גבוה!\n\n"
                     f"שימוש: {stats['rss_mb']:.1f} MB\n"
                     f"אחוז: {stats['percent']:.1f}%\n"
                     f"עלייה: +{stats['increase_mb']:.1f} MB"
            )

memory_monitor = MemoryMonitor(alert_threshold_mb=500)

# Background job to check memory
async def check_memory_job(context):
    await memory_monitor.check_and_alert(
        context.bot,
        admin_chat_id=ADMIN_CHAT_ID
    )

# Add job every 5 minutes
job_queue.run_repeating(check_memory_job, interval=300)
```

### Slow Request Detection
**למה זה שימושי:** מעקב אחר בקשות שלוקחות הרבה זמן - שיפור חוויית משתמש

```python
class RequestTimer:
    def __init__(self, slow_threshold=2.0):
        self.slow_threshold = slow_threshold
        self.slow_requests = []
    
    def time_request(self, func):
        """Decorator to time request handlers"""
        @wraps(func)
        async def wrapper(update, context):
            start = time.time()
            user_id = update.effective_user.id if update.effective_user else 'unknown'
            
            try:
                result = await func(update, context)
                duration = time.time() - start
                
                if duration > self.slow_threshold:
                    self.slow_requests.append({
                        'handler': func.__name__,
                        'user_id': user_id,
                        'duration': duration,
                        'timestamp': datetime.now()
                    })
                    
                    print(f"🐌 Slow request: {func.__name__} "
                          f"by user {user_id} took {duration:.2f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start
                print(f"❌ Request failed after {duration:.2f}s: {e}")
                raise
        
        return wrapper

timer = RequestTimer(slow_threshold=1.5)

# Apply to handlers
@timer.time_request
async def search_command(update, context):
    query = ' '.join(context.args)
    results = await db.search(query)  # Potentially slow
    await update.message.reply_text(f"נמצאו {len(results)} תוצאות")

# Get statistics
async def request_stats(update, context):
    if not timer.slow_requests:
        await update.message.reply_text("✅ כל הבקשות מהירות")
        return
    
    # Group by handler
    from collections import Counter
    handlers = Counter(r['handler'] for r in timer.slow_requests)
    
    text = "🐌 הבקשות האיטיות:\n\n"
    for handler, count in handlers.most_common(5):
        text += f"• {handler}: {count} פעמים\n"
    
    await update.message.reply_text(text)
```

### Performance Profiling Decorator
**למה זה שימושי:** פרופיילינג מפורט של פונקציות - מציאת צווארי בקבוק

```python
import cProfile
import pstats
from io import StringIO

def profile_performance(func):
    """Profile function performance and print results"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = await func(*args, **kwargs)
        
        profiler.disable()
        
        # Get stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10
        
        print(f"\n{'='*50}")
        print(f"Profile for {func.__name__}")
        print('='*50)
        print(s.getvalue())
        
        return result
    return wrapper

# Usage on complex functions
@profile_performance
async def complex_report_generation(user_id):
    # Multiple slow operations
    user = await db.users.find_one({'_id': user_id})
    posts = await db.posts.find({'user_id': user_id}).to_list()
    stats = await calculate_statistics(posts)
    report = await generate_pdf(user, stats)
    return report
```

---

## 22. Webhook vs Polling Patterns

### Webhook Setup with Security
**למה זה שימושי:** הגדרת webhook מאובטח עם אימות

```python
from fastapi import FastAPI, Request, HTTPException
from telegram import Update
import hmac
import hashlib

app = FastAPI()

def verify_telegram_webhook(token: str, secret_token: str, request_secret: str):
    """Verify webhook request is from Telegram"""
    if secret_token != request_secret:
        return False
    return True

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    # Verify secret token
    secret = request.headers.get('X-Telegram-Bot-Api-Secret-Token')
    
    if not verify_telegram_webhook(BOT_TOKEN, WEBHOOK_SECRET, secret):
        raise HTTPException(status_code=403, detail="Invalid secret token")
    
    # Process update
    json_data = await request.json()
    update = Update.de_json(json_data, bot)
    
    await application.process_update(update)
    
    return {"ok": True}

# Set webhook
async def setup_webhook():
    webhook_url = f"https://your-domain.com/webhook/{BOT_TOKEN}"
    
    await bot.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET,
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True
    )
    
    print(f"✅ Webhook set: {webhook_url}")

# Remove webhook
async def remove_webhook():
    await bot.delete_webhook(drop_pending_updates=True)
    print("✅ Webhook removed")
```

### Smart Polling with Offset Management
**למה זה שימושי:** polling יעיל שלא מפספס עדכונים ולא מעבד אותם פעמיים

```python
class SmartPoller:
    def __init__(self, bot):
        self.bot = bot
        self.offset = 0
        self.running = False
    
    async def start(self):
        """Start polling with offset management"""
        self.running = True
        
        # Load last offset from storage
        self.offset = await redis.get('polling_offset') or 0
        
        print(f"🔄 Starting polling from offset {self.offset}")
        
        while self.running:
            try:
                # Get updates
                updates = await self.bot.get_updates(
                    offset=self.offset,
                    timeout=30,
                    allowed_updates=["message", "callback_query"]
                )
                
                for update in updates:
                    # Process update
                    await application.process_update(update)
                    
                    # Update offset
                    self.offset = update.update_id + 1
                    
                    # Save offset periodically
                    await redis.set('polling_offset', self.offset)
                
            except Exception as e:
                print(f"❌ Polling error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def stop(self):
        """Stop polling gracefully"""
        self.running = False
        await redis.set('polling_offset', self.offset)
        print(f"✅ Polling stopped at offset {self.offset}")

poller = SmartPoller(bot)
```

### Hybrid Approach (Webhook with Polling Fallback)
**למה זה שימושי:** מעבר אוטומטי ל-polling אם webhook נופל

```python
class HybridUpdater:
    def __init__(self, bot, webhook_url):
        self.bot = bot
        self.webhook_url = webhook_url
        self.mode = 'webhook'
        self.poller = SmartPoller(bot)
    
    async def start(self):
        """Start in webhook mode with fallback"""
        try:
            await self.setup_webhook()
            self.mode = 'webhook'
            print("✅ Running in webhook mode")
            
            # Monitor webhook health
            asyncio.create_task(self.monitor_webhook_health())
            
        except Exception as e:
            print(f"❌ Webhook failed: {e}")
            await self.fallback_to_polling()
    
    async def setup_webhook(self):
        """Setup webhook"""
        await self.bot.set_webhook(
            url=self.webhook_url,
            secret_token=WEBHOOK_SECRET
        )
        
        # Test webhook
        webhook_info = await self.bot.get_webhook_info()
        if webhook_info.url != self.webhook_url:
            raise Exception("Webhook URL mismatch")
    
    async def fallback_to_polling(self):
        """Switch to polling mode"""
        print("⚠️  Falling back to polling mode")
        
        await self.bot.delete_webhook()
        self.mode = 'polling'
        await self.poller.start()
    
    async def monitor_webhook_health(self):
        """Check webhook health periodically"""
        while self.mode == 'webhook':
            await asyncio.sleep(300)  # Check every 5 minutes
            
            try:
                info = await self.bot.get_webhook_info()
                
                # Check for errors
                if info.last_error_date:
                    error_age = time.time() - info.last_error_date
                    
                    if error_age < 600:  # Errors in last 10 minutes
                        print(f"⚠️  Webhook errors detected: {info.last_error_message}")
                        await self.fallback_to_polling()
                        break
            
            except Exception as e:
                print(f"❌ Health check failed: {e}")
                await self.fallback_to_polling()
                break

updater = HybridUpdater(bot, WEBHOOK_URL)
```

---

## 23. Telegram Bot Commands Menu

### Dynamic Command Registration
**למה זה שימושי:** רישום פקודות דינמי עם תיאורים שמופיעים בתפריט הבוט

```python
class CommandRegistry:
    def __init__(self, bot):
        self.bot = bot
        self.commands = {}
    
    def register(self, command, description, handler, scope='default'):
        """Register command with description"""
        self.commands[command] = {
            'description': description,
            'handler': handler,
            'scope': scope
        }
    
    async def update_bot_menu(self):
        """Update Telegram bot commands menu"""
        from telegram import BotCommand
        
        # Default commands (for all users)
        default_commands = [
            BotCommand(cmd, info['description'])
            for cmd, info in self.commands.items()
            if info['scope'] == 'default'
        ]
        
        await self.bot.set_my_commands(default_commands)
        
        # Admin commands
        admin_commands = default_commands + [
            BotCommand(cmd, info['description'])
            for cmd, info in self.commands.items()
            if info['scope'] == 'admin'
        ]
        
        from telegram import BotCommandScopeChat
        
        for admin_id in ADMIN_IDS:
            await self.bot.set_my_commands(
                admin_commands,
                scope=BotCommandScopeChat(admin_id)
            )
    
    def get_handler(self, command):
        """Get handler for command"""
        return self.commands.get(command, {}).get('handler')

# Usage
registry = CommandRegistry(bot)

# Register commands
registry.register('start', 'התחל שיחה', start_handler, scope='default')
registry.register('help', 'עזרה', help_handler, scope='default')
registry.register('stats', 'סטטיסטיקות', stats_handler, scope='admin')
registry.register('broadcast', 'שלח לכולם', broadcast_handler, scope='admin')

# Update bot menu
await registry.update_bot_menu()
```

### Command Aliases System
**למה זה שימושי:** מספר שמות לאותה פקודה (כמו /help ו-/h)

```python
class CommandAliases:
    def __init__(self):
        self.aliases = {}
        self.handlers = {}
    
    def register(self, main_command, aliases, handler):
        """Register command with aliases"""
        # Store main handler
        self.handlers[main_command] = handler
        
        # Map aliases to main command
        for alias in aliases:
            self.aliases[alias] = main_command
    
    def get_handler(self, command):
        """Get handler for command or alias"""
        # Check if it's an alias
        main_command = self.aliases.get(command, command)
        return self.handlers.get(main_command)
    
    def setup_handlers(self, application):
        """Setup all handlers in application"""
        for command, handler in self.handlers.items():
            # Get all aliases for this command
            all_commands = [command] + [
                alias for alias, main in self.aliases.items()
                if main == command
            ]
            
            # Register with all aliases
            application.add_handler(
                CommandHandler(all_commands, handler)
            )

# Usage
aliases = CommandAliases()

aliases.register('help', ['h', 'עזרה'], help_handler)
aliases.register('stats', ['s', 'statistics', 'סטטיסטיקות'], stats_handler)
aliases.register('settings', ['config', 'הגדרות'], settings_handler)

aliases.setup_handlers(application)
```

### Grouped Commands Menu
**למה זה שימושי:** ארגון פקודות לקטגוריות - קל למשתמש למצוא

```python
class GroupedCommands:
    def __init__(self):
        self.groups = {
            'general': {'title': '📱 כללי', 'commands': []},
            'files': {'title': '📁 קבצים', 'commands': []},
            'admin': {'title': '👑 אדמין', 'commands': []},
        }
    
    def add_command(self, group, command, description):
        """Add command to group"""
        if group in self.groups:
            self.groups[group]['commands'].append({
                'command': command,
                'description': description
            })
    
    def get_help_text(self, user_role='user'):
        """Generate formatted help text"""
        text = "📚 רשימת פקודות\n\n"
        
        for group_key, group in self.groups.items():
            # Skip admin commands for regular users
            if group_key == 'admin' and user_role != 'admin':
                continue
            
            text += f"{group['title']}\n"
            
            for cmd in group['commands']:
                text += f"/{cmd['command']} - {cmd['description']}\n"
            
            text += "\n"
        
        return text

# Usage
commands = GroupedCommands()

# General commands
commands.add_command('general', 'start', 'התחל שיחה')
commands.add_command('general', 'help', 'הצג עזרה')
commands.add_command('general', 'settings', 'הגדרות')

# File commands
commands.add_command('files', 'upload', 'העלה קובץ')
commands.add_command('files', 'list', 'רשימת קבצים')
commands.add_command('files', 'search', 'חפש קובץ')

# Admin commands
commands.add_command('admin', 'stats', 'סטטיסטיקות')
commands.add_command('admin', 'broadcast', 'שלח לכולם')

# Handler
async def help_command(update, context):
    user_role = get_user_role(update.effective_user.id)
    help_text = commands.get_help_text(user_role)
    await update.message.reply_text(help_text)
```

---

## 24. Advanced Error Recovery

### Dead Letter Queue for Failed Messages
**למה זה שימושי:** שמירת הודעות שנכשלו לטיפול מאוחר יותר

```python
class DeadLetterQueue:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_key = 'dlq:messages'
    
    async def add(self, update, error, retry_count=0):
        """Add failed message to DLQ"""
        dlq_item = {
            'update_id': update.update_id,
            'user_id': update.effective_user.id if update.effective_user else None,
            'message': update.message.to_dict() if update.message else {},
            'error': str(error),
            'retry_count': retry_count,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.redis.lpush(
            self.queue_key,
            json.dumps(dlq_item)
        )
        
        print(f"📮 Added to DLQ: update {update.update_id}")
    
    async def get_all(self, limit=100):
        """Get all items from DLQ"""
        items = await self.redis.lrange(self.queue_key, 0, limit - 1)
        return [json.loads(item) for item in items]
    
    async def retry_failed(self, bot, max_retries=3):
        """Retry processing failed messages"""
        items = await self.get_all()
        
        for item in items:
            if item['retry_count'] >= max_retries:
                continue  # Skip if max retries reached
            
            try:
                # Reconstruct update
                update = Update.de_json(item['message'], bot)
                
                # Try processing again
                await application.process_update(update)
                
                # Remove from DLQ if successful
                await self.redis.lrem(self.queue_key, 1, json.dumps(item))
                print(f"✅ Recovered update {item['update_id']}")
                
            except Exception as e:
                # Increment retry count
                item['retry_count'] += 1
                print(f"❌ Retry failed for {item['update_id']}: {e}")

dlq = DeadLetterQueue(redis_client)

# Error handler
async def error_handler(update, context):
    error = context.error
    
    # Log error
    logger.error(f"Update {update} caused error {error}")
    
    # Add to DLQ
    await dlq.add(update, error)
    
    # Notify user
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ אירעה שגיאה. ננסה שוב מאוחר יותר."
        )

application.add_error_handler(error_handler)

# Background job to retry
async def retry_dlq_job(context):
    await dlq.retry_failed(context.bot)

job_queue.run_repeating(retry_dlq_job, interval=300)  # Every 5 minutes
```

### Error Notification System
**למה זה שימושי:** התראות מיידיות לאדמין על שגיאות קריטיות

```python
class ErrorNotifier:
    def __init__(self, bot, admin_chat_id):
        self.bot = bot
        self.admin_chat_id = admin_chat_id
        self.error_counts = {}
        self.last_notification = {}
    
    async def notify(self, error, update=None, severity='error'):
        """Send error notification to admin"""
        error_key = f"{type(error).__name__}:{str(error)[:50]}"
        
        # Count errors
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Rate limit notifications (max 1 per minute per error type)
        last_notif = self.last_notification.get(error_key, 0)
        if time.time() - last_notif < 60:
            return
        
        self.last_notification[error_key] = time.time()
        
        # Build message
        emoji = {'critical': '🔥', 'error': '❌', 'warning': '⚠️'}[severity]
        
        text = f"{emoji} <b>Error Alert</b>\n\n"
        text += f"<b>Type:</b> {type(error).__name__}\n"
        text += f"<b>Message:</b> {str(error)[:200]}\n"
        text += f"<b>Count:</b> {self.error_counts[error_key]} times\n"
        
        if update:
            text += f"\n<b>User:</b> {update.effective_user.id if update.effective_user else 'Unknown'}\n"
            text += f"<b>Update ID:</b> {update.update_id}\n"
        
        # Add traceback for critical errors
        if severity == 'critical':
            import traceback
            tb = ''.join(traceback.format_tb(error.__traceback__)[-3:])
            text += f"\n<code>{tb[:500]}</code>"
        
        await self.bot.send_message(
            chat_id=self.admin_chat_id,
            text=text,
            parse_mode='HTML'
        )

notifier = ErrorNotifier(bot, ADMIN_CHAT_ID)

async def error_handler_with_notification(update, context):
    error = context.error
    
    # Determine severity
    if isinstance(error, (NetworkError, TimedOut)):
        severity = 'warning'
    elif isinstance(error, Unauthorized):
        severity = 'error'
    else:
        severity = 'critical'
    
    # Notify admin
    await notifier.notify(error, update, severity)
```

### Automatic Recovery Strategies
**למה זה שימושי:** התאוששות אוטומטית משגיאות נפוצות

```python
class AutoRecovery:
    def __init__(self, bot):
        self.bot = bot
        self.recovery_strategies = {
            'NetworkError': self.recover_network_error,
            'TimedOut': self.recover_timeout,
            'RetryAfter': self.recover_rate_limit,
        }
    
    async def try_recover(self, error, func, *args, **kwargs):
        """Try to recover from error and retry"""
        error_type = type(error).__name__
        
        recovery_func = self.recovery_strategies.get(error_type)
        
        if recovery_func:
            print(f"🔄 Attempting recovery for {error_type}")
            
            success = await recovery_func(error)
            
            if success:
                # Retry original function
                return await func(*args, **kwargs)
        
        # No recovery possible
        raise error
    
    async def recover_network_error(self, error):
        """Recover from network errors"""
        print("🌐 Network error, waiting 5s...")
        await asyncio.sleep(5)
        return True
    
    async def recover_timeout(self, error):
        """Recover from timeout"""
        print("⏱ Timeout, retrying...")
        await asyncio.sleep(2)
        return True
    
    async def recover_rate_limit(self, error):
        """Recover from rate limiting"""
        retry_after = getattr(error, 'retry_after', 60)
        print(f"⏸ Rate limited, waiting {retry_after}s...")
        await asyncio.sleep(retry_after)
        return True

recovery = AutoRecovery(bot)

# Decorator for auto-recovery
def with_auto_recovery(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return await recovery.try_recover(e, func, *args, **kwargs)
    return wrapper

# Usage
@with_auto_recovery
async def send_message_reliable(chat_id, text):
    await bot.send_message(chat_id=chat_id, text=text)
```

### Error Aggregation & Reporting
**למה זה שימושי:** דוח מרוכז של כל השגיאות - זיהוי בעיות מתמשכות

```python
class ErrorAggregator:
    def __init__(self):
        self.errors = []
        self.start_time = datetime.now()
    
    def record(self, error, context=None):
        """Record an error"""
        self.errors.append({
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'timestamp': datetime.now()
        })
    
    def get_report(self):
        """Generate error report"""
        if not self.errors:
            return "✅ No errors recorded"
        
        # Count by type
        from collections import Counter
        error_types = Counter(e['type'] for e in self.errors)
        
        # Calculate uptime
        uptime = datetime.now() - self.start_time
        
        report = f"📊 <b>Error Report</b>\n\n"
        report += f"⏱ Uptime: {uptime}\n"
        report += f"❌ Total Errors: {len(self.errors)}\n\n"
        
        report += "<b>By Type:</b>\n"
        for error_type, count in error_types.most_common(10):
            report += f"• {error_type}: {count}\n"
        
        # Recent errors
        report += f"\n<b>Recent Errors (last 5):</b>\n"
        for error in self.errors[-5:]:
            report += f"• [{error['timestamp']:%H:%M:%S}] {error['type']}: {error['message'][:50]}\n"
        
        return report
    
    def clear(self):
        """Clear recorded errors"""
        self.errors.clear()

aggregator = ErrorAggregator()

# Record errors
async def error_handler_with_aggregation(update, context):
    error = context.error
    aggregator.record(error, context={'update_id': update.update_id if update else None})

# Daily report
async def daily_error_report(context):
    report = aggregator.get_report()
    
    await context.bot.send_message(
        chat_id=ADMIN_CHAT_ID,
        text=report,
        parse_mode='HTML'
    )
    
    # Clear after reporting
    aggregator.clear()

# Schedule daily at 9 AM
job_queue.run_daily(daily_error_report, time=time(hour=9, minute=0))
```

