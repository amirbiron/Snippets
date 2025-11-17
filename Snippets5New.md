# Code Snippets Library for Telegram Bot Developers

## 1. Webhook Handling & Security

### Webhook Signature Verification
**מטרה:** לוודא שהבקשות באמת מגיעות מטלגרם ולא מתוקף
```python
import hmac
import hashlib
from fastapi import Header, HTTPException

async def verify_telegram_webhook(
    request_body: bytes,
    x_telegram_bot_api_secret_token: str = Header(None)
) -> bool:
    """Verify that webhook request comes from Telegram"""
    expected_token = os.getenv("WEBHOOK_SECRET_TOKEN")
    
    if not x_telegram_bot_api_secret_token:
        raise HTTPException(401, "Missing secret token")
    
    if not hmac.compare_digest(
        x_telegram_bot_api_secret_token, 
        expected_token
    ):
        raise HTTPException(403, "Invalid secret token")
    
    return True
```

### Webhook Retry Mechanism
**מטרה:** לטפל בכשלים זמניים ולוודא שאף עדכון לא אובד
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def process_webhook_update(update: dict):
    """Process update with automatic retry on failure"""
    try:
        await handle_update(update)
    except Exception as e:
        logger.error(f"Failed processing update {update.get('update_id')}: {e}")
        raise  # Will trigger retry
```

### Webhook Health Monitoring
**מטרה:** לעקוב אחרי בריאות ה-webhook ולזהות בעיות מוקדם
```python
from datetime import datetime, timedelta
from collections import deque

class WebhookHealthMonitor:
    def __init__(self, window_minutes=5):
        self.requests = deque(maxlen=1000)
        self.errors = deque(maxlen=1000)
        self.window = timedelta(minutes=window_minutes)
    
    def record_request(self, success: bool = True):
        now = datetime.now()
        self.requests.append(now)
        if not success:
            self.errors.append(now)
    
    def get_health_status(self) -> dict:
        now = datetime.now()
        cutoff = now - self.window
        
        recent_requests = sum(1 for t in self.requests if t > cutoff)
        recent_errors = sum(1 for t in self.errors if t > cutoff)
        
        error_rate = recent_errors / recent_requests if recent_requests else 0
        
        return {
            "status": "healthy" if error_rate < 0.05 else "degraded",
            "requests_per_minute": recent_requests / self.window.seconds * 60,
            "error_rate": error_rate,
            "last_request": max(self.requests) if self.requests else None
        }
```

---

## 2. Rate Limiting מתקדם

### Token Bucket Algorithm
**מטרה:** למנוע spam והצפה תוך שמירה על חווית משתמש טובה
```python
import time
from collections import defaultdict

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.buckets = defaultdict(lambda: {
            'tokens': capacity,
            'last_refill': time.time()
        })
    
    def consume(self, key: str, tokens: int = 1) -> bool:
        bucket = self.buckets[key]
        now = time.time()
        
        # Refill tokens
        time_passed = now - bucket['last_refill']
        bucket['tokens'] = min(
            self.capacity,
            bucket['tokens'] + time_passed * self.refill_rate
        )
        bucket['last_refill'] = now
        
        # Try to consume
        if bucket['tokens'] >= tokens:
            bucket['tokens'] -= tokens
            return True
        return False
```

### Rate Limiting Per User/Chat
**מטרה:** להגביל כל משתמש/צ'אט בנפרד ולא את כל הבוט
```python
from functools import wraps
from telegram import Update

class UserRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.bucket = TokenBucket(
            capacity=max_requests,
            refill_rate=max_requests / window_seconds
        )
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(update: Update, context):
            user_id = update.effective_user.id
            
            if not self.bucket.consume(f"user:{user_id}"):
                await update.message.reply_text(
                    "⏳ Too many requests. Please wait a moment."
                )
                return
            
            return await func(update, context)
        return wrapper

# Usage
@UserRateLimiter(max_requests=10, window_seconds=60)
async def handle_command(update: Update, context):
    await update.message.reply_text("Processing...")
```

### Distributed Rate Limiting with Redis
**מטרה:** לשתף מגבלות בין מספר שרתים/workers
```python
import redis.asyncio as redis
from datetime import timedelta

class RedisRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(
        self, 
        key: str, 
        max_requests: int, 
        window: timedelta
    ) -> bool:
        pipe = self.redis.pipeline()
        now = int(time.time())
        window_start = now - int(window.total_seconds())
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current requests
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, window)
        
        results = await pipe.execute()
        request_count = results[1]
        
        return request_count < max_requests

# Usage
limiter = RedisRateLimiter(redis_client)
if await limiter.is_allowed(f"user:{user_id}", 20, timedelta(minutes=1)):
    # Process request
    pass
```

---

## 3. Retry & Circuit Breaker Patterns

### Exponential Backoff Retry
**מטרה:** לנסות שוב בצורה חכמה כשיש כשלון זמני
```python
import asyncio
import random
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    *args, **kwargs
) -> T:
    """Retry with exponential backoff and jitter"""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Calculate delay with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            wait_time = delay + jitter
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {wait_time:.2f}s..."
            )
            await asyncio.sleep(wait_time)
```

### Circuit Breaker Pattern
**מטרה:** להפסיק לנסות כשהשירות נופל, לחסוך משאבים
```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(
        self, 
        failure_threshold: int = 5,
        timeout: timedelta = timedelta(seconds=60),
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.opened_at > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = datetime.now()
```

### Retry Decorator with Jitter
**מטרה:** דקורטור נוח לשימוש חוזר בכל הפרויקט
```python
from functools import wraps
import asyncio
import random

def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.3)
                    await asyncio.sleep(delay + jitter)
                    
                    logger.info(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}")
        return wrapper
    return decorator

# Usage
@async_retry(max_attempts=3, base_delay=2.0)
async def send_telegram_message(chat_id, text):
    await bot.send_message(chat_id, text)
```

---

## 4. Telegram Inline Queries

### Inline Query Handler with Caching
**מטרה:** לשפר ביצועים ולהפחית עומס על השרת/API
```python
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import InlineQueryHandler
from functools import lru_cache
from hashlib import md5

class CachedInlineQueryHandler:
    def __init__(self, cache_ttl: int = 300):
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    @staticmethod
    def _cache_key(query: str) -> str:
        return md5(query.encode()).hexdigest()
    
    async def handle_inline_query(self, update, context):
        query = update.inline_query.query
        
        if not query:
            return
        
        # Check cache
        cache_key = self._cache_key(query)
        cached = self.cache.get(cache_key)
        
        if cached and time.time() - cached['time'] < self.cache_ttl:
            results = cached['results']
        else:
            # Perform search
            results = await self.search_content(query)
            self.cache[cache_key] = {
                'results': results,
                'time': time.time()
            }
        
        await update.inline_query.answer(results, cache_time=self.cache_ttl)
    
    async def search_content(self, query: str) -> list:
        """Override this with your search logic"""
        return [
            InlineQueryResultArticle(
                id=str(i),
                title=f"Result {i}",
                input_message_content=InputTextMessageContent(f"Content: {query}")
            )
            for i in range(5)
        ]
```

### Inline Result Pagination
**מטרה:** לאפשר גלילה דרך תוצאות רבות ב-inline mode
```python
from telegram import InlineQueryResultsButton, InlineQueryResultArticle
from uuid import uuid4

class InlineQueryPaginator:
    def __init__(self, page_size: int = 10):
        self.page_size = page_size
        self.results_cache = {}  # Store full results temporarily
    
    async def handle_paginated_query(self, update, context):
        query = update.inline_query.query
        offset = update.inline_query.offset or "0"
        
        # Get or generate all results
        session_id = str(uuid4())
        if offset == "0":
            all_results = await self.fetch_all_results(query)
            self.results_cache[session_id] = all_results
        else:
            session_id = offset.split(":")[0]
            all_results = self.results_cache.get(session_id, [])
        
        # Parse offset
        start = int(offset.split(":")[-1]) if ":" in offset else 0
        end = start + self.page_size
        
        # Slice results
        page_results = all_results[start:end]
        
        # Calculate next offset
        next_offset = f"{session_id}:{end}" if end < len(all_results) else ""
        
        await update.inline_query.answer(
            page_results,
            next_offset=next_offset,
            cache_time=10
        )
    
    async def fetch_all_results(self, query: str) -> list:
        """Fetch all matching results"""
        # Your search logic here
        return []
```

### Inline Query Rate Limiting
**מטרה:** למנוע שימוש לרעה ב-inline queries
```python
from telegram.ext import InlineQueryHandler
from collections import defaultdict
import time

class RateLimitedInlineHandler:
    def __init__(self, max_queries: int = 30, window: int = 60):
        self.max_queries = max_queries
        self.window = window
        self.user_queries = defaultdict(list)
    
    async def handle_inline_query(self, update, context):
        user_id = update.inline_query.from_user.id
        now = time.time()
        
        # Clean old entries
        self.user_queries[user_id] = [
            t for t in self.user_queries[user_id]
            if now - t < self.window
        ]
        
        # Check limit
        if len(self.user_queries[user_id]) >= self.max_queries:
            await update.inline_query.answer(
                [],
                switch_pm_text="⚠️ Rate limit exceeded",
                switch_pm_parameter="rate_limit"
            )
            return
        
        # Record query
        self.user_queries[user_id].append(now)
        
        # Process normally
        results = await self.process_query(update.inline_query.query)
        await update.inline_query.answer(results)
```

---

## 5. Message Scheduling & Delayed Tasks

### Schedule Messages for Future
**מטרה:** לשלוח הודעות בזמן מוגדר מראש
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from datetime import datetime

class MessageScheduler:
    def __init__(self, bot):
        self.bot = bot
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
    
    def schedule_message(
        self, 
        chat_id: int, 
        text: str, 
        send_at: datetime,
        job_id: str = None
    ) -> str:
        """Schedule a message to be sent at specific time"""
        job_id = job_id or f"msg_{chat_id}_{int(send_at.timestamp())}"
        
        self.scheduler.add_job(
            self._send_message,
            trigger=DateTrigger(run_date=send_at),
            args=[chat_id, text],
            id=job_id,
            replace_existing=True
        )
        
        return job_id
    
    async def _send_message(self, chat_id: int, text: str):
        try:
            await self.bot.send_message(chat_id, text)
        except Exception as e:
            logger.error(f"Failed to send scheduled message: {e}")
    
    def cancel_scheduled_message(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
            return True
        except:
            return False

# Usage
scheduler = MessageScheduler(bot)
scheduler.schedule_message(
    chat_id=123456,
    text="Reminder: Meeting in 10 minutes!",
    send_at=datetime.now() + timedelta(hours=1)
)
```

### Delayed Job Execution
**מטרה:** להריץ משימות לאחר השהייה מוגדרת
```python
import asyncio
from typing import Callable, Any

class DelayedJobQueue:
    def __init__(self):
        self.tasks = {}
    
    async def schedule_delayed(
        self,
        job_id: str,
        func: Callable,
        delay_seconds: float,
        *args,
        **kwargs
    ):
        """Execute function after delay"""
        # Cancel existing job with same ID
        if job_id in self.tasks:
            self.tasks[job_id].cancel()
        
        async def delayed_execution():
            await asyncio.sleep(delay_seconds)
            try:
                await func(*args, **kwargs)
            finally:
                self.tasks.pop(job_id, None)
        
        task = asyncio.create_task(delayed_execution())
        self.tasks[job_id] = task
        return task
    
    def cancel_job(self, job_id: str) -> bool:
        if job_id in self.tasks:
            self.tasks[job_id].cancel()
            self.tasks.pop(job_id)
            return True
        return False

# Usage
queue = DelayedJobQueue()
await queue.schedule_delayed(
    job_id=f"delete_msg_{chat_id}_{msg_id}",
    func=bot.delete_message,
    delay_seconds=300,  # 5 minutes
    chat_id=chat_id,
    message_id=msg_id
)
```

### Recurring Tasks
**מטרה:** להריץ משימות חוזרות במרווחי זמן קבועים
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

class RecurringTaskManager:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
    
    def add_daily_task(
        self, 
        func: Callable, 
        hour: int, 
        minute: int = 0,
        job_id: str = None
    ):
        """Run task daily at specific time"""
        self.scheduler.add_job(
            func,
            trigger=CronTrigger(hour=hour, minute=minute),
            id=job_id,
            replace_existing=True
        )
    
    def add_interval_task(
        self,
        func: Callable,
        minutes: int = None,
        hours: int = None,
        job_id: str = None
    ):
        """Run task at regular intervals"""
        self.scheduler.add_job(
            func,
            trigger=IntervalTrigger(minutes=minutes, hours=hours),
            id=job_id,
            replace_existing=True
        )
    
    def remove_task(self, job_id: str):
        self.scheduler.remove_job(job_id)

# Usage
tasks = RecurringTaskManager()

# Send daily summary at 9 AM
tasks.add_daily_task(
    send_daily_summary,
    hour=9,
    minute=0,
    job_id="daily_summary"
)

# Check for updates every 30 minutes
tasks.add_interval_task(
    check_updates,
    minutes=30,
    job_id="update_checker"
)
```

---

## 6. Database Migrations

### Alembic Migration Patterns
**מטרה:** ניהול שינויי מבנה DB באופן בטוח ומבוקר
```python
# alembic/env.py helper
from alembic import context
from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig

def run_migrations_online():
    """Run migrations in 'online' mode with connection pooling"""
    connectable = engine_from_config(
        context.config.get_section(context.config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Detect column type changes
            compare_server_default=True,  # Detect default changes
            include_schemas=True,  # Support multiple schemas
        )

        with context.begin_transaction():
            context.run_migrations()

# Migration file pattern for adding column with data
def upgrade():
    # Add column as nullable first
    op.add_column('users', sa.Column('status', sa.String(20), nullable=True))
    
    # Populate existing rows
    op.execute("UPDATE users SET status = 'active' WHERE status IS NULL")
    
    # Make it non-nullable
    op.alter_column('users', 'status', nullable=False)

def downgrade():
    op.drop_column('users', 'status')
```

### Data Migration Helpers
**מטרה:** להעביר ולהמיר דאטה בצורה בטוחה במהלך migration
```python
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column

def batch_update_data(
    table_name: str,
    update_column: str,
    condition_column: str,
    update_func: Callable,
    batch_size: int = 1000
):
    """Update data in batches to avoid locking table"""
    conn = op.get_bind()
    
    # Create table reference
    t = table(
        table_name,
        column('id', sa.Integer),
        column(condition_column),
        column(update_column)
    )
    
    offset = 0
    while True:
        # Fetch batch
        rows = conn.execute(
            sa.select(t.c.id, t.c[condition_column])
            .limit(batch_size)
            .offset(offset)
        ).fetchall()
        
        if not rows:
            break
        
        # Update batch
        for row in rows:
            new_value = update_func(row[1])
            conn.execute(
                t.update()
                .where(t.c.id == row[0])
                .values({update_column: new_value})
            )
        
        offset += batch_size

# Usage in migration
def upgrade():
    op.add_column('messages', sa.Column('processed_text', sa.Text))
    
    batch_update_data(
        'messages',
        update_column='processed_text',
        condition_column='raw_text',
        update_func=lambda text: text.strip().lower()
    )
```

### Rollback Strategies
**מטרה:** לוודא שאפשר לחזור אחורה בצורה בטוחה
```python
def safe_migration_with_backup():
    """Pattern for safe migration with backup table"""
    # Create backup table
    op.execute("""
        CREATE TABLE users_backup AS 
        SELECT * FROM users
    """)
    
    try:
        # Perform migration
        op.add_column('users', sa.Column('new_field', sa.String(100)))
        op.execute("UPDATE users SET new_field = old_field || '_processed'")
        op.drop_column('users', 'old_field')
        
        # Drop backup if successful
        op.execute("DROP TABLE users_backup")
    except Exception as e:
        # Restore from backup
        op.execute("DROP TABLE users")
        op.execute("ALTER TABLE users_backup RENAME TO users")
        raise

def downgrade():
    """Always include meaningful downgrade"""
    # Re-add old column
    op.add_column('users', sa.Column('old_field', sa.String(100)))
    
    # Reverse data transformation
    op.execute("""
        UPDATE users 
        SET old_field = REPLACE(new_field, '_processed', '')
    """)
    
    # Remove new column
    op.drop_column('users', 'new_field')
```

---

## 7. API Client Patterns

### Async HTTP Client with Retry
**מטרה:** client אמין לקריאות API עם טיפול אוטומטי בשגיאות
```python
import aiohttp
from typing import Optional, Dict, Any
import asyncio

class ResilientAPIClient:
    def __init__(
        self, 
        base_url: str, 
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    async def request(
        self, 
        method: str, 
        endpoint: str,
        **kwargs
    ) -> Dict[Any, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait = 2 ** attempt
                logger.warning(f"Request failed, retry {attempt + 1} in {wait}s")
                await asyncio.sleep(wait)
        
    async def get(self, endpoint: str, **kwargs):
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs):
        return await self.request("POST", endpoint, **kwargs)

# Usage
async with ResilientAPIClient("https://api.example.com") as client:
    data = await client.get("/users/123")
```

### Request/Response Logging
**מטרה:** לעקוב אחרי כל הקריאות ל-API לצורך debug ואנליזה
```python
import json
import logging
from functools import wraps
from time import time

class APILogger:
    def __init__(self, logger_name: str = "api_client"):
        self.logger = logging.getLogger(logger_name)
    
    def log_request_response(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time()
            
            # Log request
            self.logger.info(
                f"API Request: {func.__name__}",
                extra={
                    "args": str(args)[:200],
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
                }
            )
            
            try:
                response = await func(*args, **kwargs)
                duration = time() - start_time
                
                # Log successful response
                self.logger.info(
                    f"API Response: {func.__name__} completed",
                    extra={
                        "duration_ms": int(duration * 1000),
                        "response_size": len(str(response))
                    }
                )
                
                return response
                
            except Exception as e:
                duration = time() - start_time
                
                # Log error
                self.logger.error(
                    f"API Error: {func.__name__} failed",
                    extra={
                        "duration_ms": int(duration * 1000),
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                raise
        
        return wrapper

# Usage
api_logger = APILogger()

class MyAPIClient:
    @api_logger.log_request_response
    async def get_user(self, user_id: int):
        # API call logic
        pass
```

### API Rate Limiting Compliance
**מטרה:** לכבד את מגבלות ה-API ולמנוע blocking
```python
import asyncio
from datetime import datetime, timedelta
from collections import deque

class APIRateLimiter:
    def __init__(self, requests_per_second: int):
        self.rate = requests_per_second
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until we can make another request"""
        async with self.lock:
            now = datetime.now()
            
            # Remove requests older than 1 second
            while self.requests and now - self.requests[0] > timedelta(seconds=1):
                self.requests.popleft()
            
            # If at limit, wait
            if len(self.requests) >= self.rate:
                sleep_time = 1 - (now - self.requests[0]).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.requests.popleft()
            
            self.requests.append(datetime.now())

class RateLimitedClient:
    def __init__(self, requests_per_second: int = 10):
        self.limiter = APIRateLimiter(requests_per_second)
    
    async def call_api(self, endpoint: str):
        await self.limiter.acquire()
        # Make actual API call
        return await self._make_request(endpoint)

# Usage with multiple requests
client = RateLimitedClient(requests_per_second=20)
tasks = [client.call_api(f"/endpoint/{i}") for i in range(100)]
results = await asyncio.gather(*tasks)  # Will respect rate limit
```

---

## 8. Decorators שימושיים

### @retry, @timeout, @cache
**מטרה:** דקורטורים מוכנים לשימוש לניהול שגיאות וביצועים
```python
import asyncio
from functools import wraps
from typing import Optional
import time

def retry(attempts: int = 3, delay: float = 1.0):
    """Retry decorator for async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == attempts - 1:
                        raise
                    await asyncio.sleep(delay * (attempt + 1))
        return wrapper
    return decorator

def timeout(seconds: float):
    """Timeout decorator for async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"{func.__name__} exceeded {seconds}s timeout"
                )
        return wrapper
    return decorator

def cache(ttl: Optional[int] = None):
    """Simple cache decorator with optional TTL"""
    cache_data = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            
            if key in cache_data:
                cached_time, cached_value = cache_data[key]
                if ttl is None or time.time() - cached_time < ttl:
                    return cached_value
            
            result = await func(*args, **kwargs)
            cache_data[key] = (time.time(), result)
            return result
        
        return wrapper
    return decorator

# Usage
@retry(attempts=3)
@timeout(10)
@cache(ttl=300)
async def fetch_user_data(user_id: int):
    # Will retry up to 3 times, timeout after 10s, cache for 5 minutes
    return await api.get(f"/users/{user_id}")
```

### @require_auth, @admin_only
**מטרה:** בקרת גישה פשוטה ויעילה
```python
from functools import wraps
from telegram import Update
from telegram.ext import ContextTypes

def require_auth(func):
    """Require user to be registered"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        user = await db.get_user(user_id)
        if not user:
            await update.message.reply_text(
                "⚠️ Please register first with /start"
            )
            return
        
        return await func(update, context)
    return wrapper

def admin_only(func):
    """Restrict command to admin users"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("⛔ Admin access required")
            return
        
        return await func(update, context)
    return wrapper

def chat_type(*allowed_types):
    """Restrict to specific chat types (private, group, channel)"""
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            chat_type = update.effective_chat.type
            
            if chat_type not in allowed_types:
                await update.message.reply_text(
                    f"This command works only in: {', '.join(allowed_types)}"
                )
                return
            
            return await func(update, context)
        return wrapper
    return decorator

# Usage
@admin_only
async def ban_user(update, context):
    # Only admins can use this
    pass

@require_auth
@chat_type("private")
async def settings(update, context):
    # Only registered users in private chat
    pass
```

### @log_execution_time
**מטרה:** לעקוב אחרי ביצועים ולזהות bottlenecks
```python
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

def log_execution_time(threshold_ms: Optional[int] = None):
    """Log execution time, warn if exceeds threshold"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                
                log_data = {
                    "function": func.__name__,
                    "duration_ms": round(duration_ms, 2)
                }
                
                if threshold_ms and duration_ms > threshold_ms:
                    logger.warning(
                        f"Slow execution: {func.__name__} took {duration_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms)",
                        extra=log_data
                    )
                else:
                    logger.debug(
                        f"{func.__name__} completed in {duration_ms:.2f}ms",
                        extra=log_data
                    )
        
        return wrapper
    return decorator

# Usage
@log_execution_time(threshold_ms=1000)
async def process_large_file(file_path: str):
    # Will warn if takes > 1 second
    pass
```

---

## 9. Telegram Payments & Invoices

### Payment Processing
**מטרה:** לקבל תשלומים דרך Telegram בצורה בטוחה
```python
from telegram import LabeledPrice, Update
from telegram.ext import ContextTypes, PreCheckoutQueryHandler

class PaymentProcessor:
    def __init__(self, provider_token: str):
        self.provider_token = provider_token
    
    async def send_invoice(
        self,
        update: Update,
        title: str,
        description: str,
        payload: str,
        amount: int,  # In smallest currency unit (cents)
        currency: str = "USD"
    ):
        """Send payment invoice to user"""
        await update.message.reply_invoice(
            title=title,
            description=description,
            payload=payload,
            provider_token=self.provider_token,
            currency=currency,
            prices=[LabeledPrice("Price", amount)],
            start_parameter="payment",
            photo_url="https://example.com/product.jpg",  # Optional
            need_name=True,
            need_email=True,
            need_shipping_address=False,
        )
    
    async def handle_pre_checkout(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Validate payment before processing"""
        query = update.pre_checkout_query
        
        # Verify payload and amount
        if not await self._validate_payment(query.invoice_payload, query.total_amount):
            await query.answer(
                ok=False,
                error_message="Payment validation failed"
            )
            return
        
        await query.answer(ok=True)
    
    async def handle_successful_payment(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Process successful payment"""
        payment = update.message.successful_payment
        
        # Log payment
        await db.save_payment({
            "user_id": update.effective_user.id,
            "amount": payment.total_amount,
            "currency": payment.currency,
            "payload": payment.invoice_payload,
            "telegram_payment_id": payment.telegram_payment_charge_id
        })
        
        # Grant access or deliver product
        await self._fulfill_order(payment.invoice_payload, update.effective_user.id)
        
        await update.message.reply_text(
            "✅ Payment successful! Thank you for your purchase."
        )

# Usage
payment_processor = PaymentProcessor(PAYMENT_PROVIDER_TOKEN)
app.add_handler(PreCheckoutQueryHandler(payment_processor.handle_pre_checkout))
```

### Invoice Generation
**מטרה:** ליצור חשבוניות מובנות עם מספר פריטים
```python
from telegram import LabeledPrice
from dataclasses import dataclass
from typing import List

@dataclass
class InvoiceItem:
    label: str
    amount: int  # In cents

class InvoiceBuilder:
    def __init__(self):
        self.items: List[InvoiceItem] = []
    
    def add_item(self, label: str, amount: int):
        """Add item to invoice"""
        self.items.append(InvoiceItem(label, amount))
        return self
    
    def add_tax(self, percentage: float):
        """Add tax based on current total"""
        subtotal = self.get_total()
        tax_amount = int(subtotal * percentage / 100)
        self.items.append(InvoiceItem(f"Tax ({percentage}%)", tax_amount))
        return self
    
    def add_discount(self, label: str, amount: int):
        """Add discount (negative amount)"""
        self.items.append(InvoiceItem(label, -amount))
        return self
    
    def get_total(self) -> int:
        return sum(item.amount for item in self.items)
    
    def build_prices(self) -> List[LabeledPrice]:
        """Convert to Telegram LabeledPrice format"""
        return [
            LabeledPrice(item.label, item.amount)
            for item in self.items
        ]

# Usage
invoice = InvoiceBuilder()
invoice.add_item("Premium Plan (1 month)", 999)  # $9.99
invoice.add_item("Extra Storage (10GB)", 299)     # $2.99
invoice.add_discount("First-time user discount", 200)  # -$2.00
invoice.add_tax(10)  # 10% tax

await bot.send_invoice(
    chat_id=user_id,
    title="Subscription Invoice",
    description="Your monthly subscription",
    payload="subscription_monthly",
    provider_token=PROVIDER_TOKEN,
    currency="USD",
    prices=invoice.build_prices()
)
```

### Payment Verification
**מטרה:** לוודא תקינות תשלומים ולמנוע הונאות
```python
import hmac
import hashlib
from typing import Optional

class PaymentVerifier:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
    
    def verify_payment_signature(
        self,
        invoice_payload: str,
        total_amount: int,
        currency: str,
        provider_payment_id: str,
        signature: str
    ) -> bool:
        """Verify payment signature from provider"""
        data = f"{invoice_payload}:{total_amount}:{currency}:{provider_payment_id}"
        expected_signature = hmac.new(
            self.bot_token.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    async def validate_payment(
        self,
        user_id: int,
        invoice_payload: str,
        amount: int
    ) -> tuple[bool, Optional[str]]:
        """Validate payment details before processing"""
        # Check if user exists
        user = await db.get_user(user_id)
        if not user:
            return False, "User not found"
        
        # Check if payment already processed
        existing = await db.get_payment_by_payload(invoice_payload)
        if existing:
            return False, "Payment already processed"
        
        # Validate amount
        expected_amount = await self._get_expected_amount(invoice_payload)
        if amount != expected_amount:
            return False, f"Amount mismatch: expected {expected_amount}, got {amount}"
        
        return True, None

# Usage in pre-checkout handler
verifier = PaymentVerifier(BOT_TOKEN)

async def pre_checkout_callback(update, context):
    query = update.pre_checkout_query
    
    valid, error = await verifier.validate_payment(
        user_id=query.from_user.id,
        invoice_payload=query.invoice_payload,
        amount=query.total_amount
    )
    
    await query.answer(ok=valid, error_message=error)
```

---

## 10. Monitoring & Metrics

### Prometheus Metrics Collection
**מטרה:** לאסוף ולחשוף מטריקות לניטור ב-Prometheus
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time

# Define metrics
message_counter = Counter(
    'bot_messages_total',
    'Total messages processed',
    ['command', 'status']
)

response_time = Histogram(
    'bot_response_seconds',
    'Response time distribution',
    ['handler']
)

active_users = Gauge(
    'bot_active_users',
    'Number of active users'
)

error_counter = Counter(
    'bot_errors_total',
    'Total errors',
    ['error_type']
)

class MetricsCollector:
    @staticmethod
    def track_message(command: str, status: str = "success"):
        """Track message processing"""
        message_counter.labels(command=command, status=status).inc()
    
    @staticmethod
    def track_response_time(handler_name: str):
        """Decorator to track handler response time"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    MetricsCollector.track_message(handler_name, "success")
                    return result
                except Exception as e:
                    MetricsCollector.track_message(handler_name, "error")
                    error_counter.labels(error_type=type(e).__name__).inc()
                    raise
                finally:
                    duration = time.time() - start
                    response_time.labels(handler=handler_name).observe(duration)
            return wrapper
        return decorator
    
    @staticmethod
    async def update_active_users():
        """Update active users gauge"""
        count = await db.count_active_users(minutes=5)
        active_users.set(count)

# Expose metrics endpoint (FastAPI)
from fastapi import FastAPI, Response
app = FastAPI()

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Usage
@MetricsCollector.track_response_time("start_command")
async def start_command(update, context):
    await update.message.reply_text("Hello!")
```

### Custom Metrics Helpers
**מטרה:** יצירת מטריקות מותאמות אישית לצרכים ספציפיים
```python
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
import json

class CustomMetrics:
    def __init__(self):
        self.metrics: Dict[str, List] = defaultdict(list)
        self.aggregations: Dict[str, float] = {}
    
    def record(self, metric_name: str, value: float, tags: dict = None):
        """Record a metric value with optional tags"""
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        })
        
        # Keep only last hour of data
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics[metric_name] = [
            m for m in self.metrics[metric_name]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
    
    def get_average(self, metric_name: str, minutes: int = 5) -> float:
        """Get average value over last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        values = [
            m["value"] for m in self.metrics.get(metric_name, [])
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        return sum(values) / len(values) if values else 0
    
    def get_percentile(self, metric_name: str, percentile: int) -> float:
        """Get percentile value (e.g., 95th percentile)"""
        values = sorted([m["value"] for m in self.metrics.get(metric_name, [])])
        if not values:
            return 0
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]
    
    def export_summary(self) -> dict:
        """Export metrics summary"""
        return {
            name: {
                "count": len(values),
                "avg": self.get_average(name, minutes=60),
                "p95": self.get_percentile(name, 95),
                "p99": self.get_percentile(name, 99)
            }
            for name, values in self.metrics.items()
        }

# Usage
metrics = CustomMetrics()

# Track custom metric
metrics.record("api_latency_ms", 45.2, tags={"endpoint": "/users"})
metrics.record("cache_hit_rate", 0.85)

# Get insights
avg_latency = metrics.get_average("api_latency_ms", minutes=5)
p95_latency = metrics.get_percentile("api_latency_ms", 95)
```

### Performance Tracking Decorators
**מטרה:** לעקוב אוטומטית אחרי ביצועים של פונקציות
```python
from functools import wraps
import time
from typing import Optional, Callable
import asyncio

class PerformanceTracker:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.thresholds = {}
    
    def track(
        self, 
        metric_name: Optional[str] = None,
        slow_threshold_ms: Optional[int] = None,
        alert_callback: Optional[Callable] = None
    ):
        """Decorator to track function performance"""
        def decorator(func):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                exception = None
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    exception = e
                    raise
                finally:
                    duration_ms = (time.time() - start) * 1000
                    
                    # Record metric
                    self.metrics.record(
                        f"{name}.duration_ms",
                        duration_ms,
                        tags={
                            "success": exception is None,
                            "error_type": type(exception).__name__ if exception else None
                        }
                    )
                    
                    # Alert if slow
                    if slow_threshold_ms and duration_ms > slow_threshold_ms:
                        logger.warning(
                            f"Slow execution: {name} took {duration_ms:.2f}ms "
                            f"(threshold: {slow_threshold_ms}ms)"
                        )
                        
                        if alert_callback:
                            asyncio.create_task(
                                alert_callback(name, duration_ms, slow_threshold_ms)
                            )
            
            return wrapper
        return decorator

# Usage
tracker = PerformanceTracker(metrics)

async def send_alert(func_name, duration, threshold):
    await bot.send_message(
        ADMIN_CHAT_ID,
        f"⚠️ Slow function: {func_name}\n"
        f"Duration: {duration:.2f}ms (threshold: {threshold}ms)"
    )

@tracker.track(slow_threshold_ms=1000, alert_callback=send_alert)
async def process_user_request(user_id: int):
    # Your code here
    pass
```

---

## 11. Environment & Secrets Management

### .env Loading with Validation
**מטרה:** לטעון הגדרות בצורה בטוחה עם ולידציה
```python
from pydantic import BaseSettings, validator, SecretStr
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Bot settings
    bot_token: SecretStr
    webhook_url: Optional[str] = None
    webhook_secret: SecretStr
    
    # Database
    database_url: SecretStr
    db_pool_size: int = 10
    
    # Redis
    redis_url: Optional[str] = None
    
    # Features
    enable_payments: bool = False
    payment_provider_token: Optional[SecretStr] = None
    
    # Admin
    admin_user_ids: List[int] = []
    
    # API Keys
    openai_api_key: Optional[SecretStr] = None
    
    @validator('bot_token')
    def validate_bot_token(cls, v):
        token = v.get_secret_value()
        if not token or ':' not in token:
            raise ValueError('Invalid bot token format')
        return v
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        if v and not v.startswith('https://'):
            raise ValueError('Webhook URL must use HTTPS')
        return v
    
    @validator('admin_user_ids', pre=True)
    def parse_admin_ids(cls, v):
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(',') if x.strip()]
        return v
    
    @validator('enable_payments')
    def validate_payment_config(cls, v, values):
        if v and not values.get('payment_provider_token'):
            raise ValueError('payment_provider_token required when payments enabled')
        return v
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

# Usage
try:
    settings = Settings()
    print("✅ Configuration loaded successfully")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    exit(1)

# Access values
BOT_TOKEN = settings.bot_token.get_secret_value()
ADMIN_IDS = settings.admin_user_ids
```

### Secrets Rotation
**מטרה:** לאפשר החלפת secrets בלי להפסיק את השירות
```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

class SecretManager:
    def __init__(self):
        self.secrets: Dict[str, dict] = {}
        self.rotation_callbacks = {}
    
    def register_secret(
        self,
        name: str,
        value: str,
        rotation_interval: Optional[timedelta] = None,
        rotation_callback: Optional[callable] = None
    ):
        """Register a secret with optional rotation"""
        self.secrets[name] = {
            "value": value,
            "created_at": datetime.now(),
            "rotation_interval": rotation_interval,
            "last_rotated": datetime.now()
        }
        
        if rotation_callback:
            self.rotation_callbacks[name] = rotation_callback
    
    def get_secret(self, name: str) -> str:
        """Get current secret value"""
        if name not in self.secrets:
            raise KeyError(f"Secret '{name}' not found")
        return self.secrets[name]["value"]
    
    async def rotate_secret(self, name: str, new_value: str):
        """Rotate a secret"""
        if name not in self.secrets:
            raise KeyError(f"Secret '{name}' not found")
        
        old_value = self.secrets[name]["value"]
        
        # Update secret
        self.secrets[name].update({
            "value": new_value,
            "last_rotated": datetime.now()
        })
        
        # Call rotation callback if registered
        if name in self.rotation_callbacks:
            await self.rotation_callbacks[name](old_value, new_value)
        
        logger.info(f"Secret '{name}' rotated successfully")
    
    async def check_rotation_needed(self):
        """Background task to check if secrets need rotation"""
        while True:
            for name, secret in self.secrets.items():
                interval = secret.get("rotation_interval")
                if not interval:
                    continue
                
                time_since_rotation = datetime.now() - secret["last_rotated"]
                if time_since_rotation > interval:
                    logger.warning(
                        f"Secret '{name}' needs rotation "
                        f"(last rotated {time_since_rotation.days} days ago)"
                    )
            
            await asyncio.sleep(3600)  # Check every hour

# Usage
secret_manager = SecretManager()

async def on_webhook_secret_rotation(old_secret, new_secret):
    # Update webhook with new secret
    await bot.set_webhook(
        url=WEBHOOK_URL,
        secret_token=new_secret
    )

secret_manager.register_secret(
    "webhook_secret",
    os.getenv("WEBHOOK_SECRET"),
    rotation_interval=timedelta(days=30),
    rotation_callback=on_webhook_secret_rotation
)
```

### Multi-Environment Config
**מטרה:** לנהל הגדרות שונות ל-dev/staging/production
```python
from enum import Enum
from pydantic import BaseSettings
from typing import Dict, Any

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class BaseConfig(BaseSettings):
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    class Config:
        env_file = '.env'

class DevelopmentConfig(BaseConfig):
    debug: bool = True
    database_url: str = "sqlite:///dev.db"
    log_level: str = "DEBUG"
    enable_sentry: bool = False

class StagingConfig(BaseConfig):
    database_url: str
    log_level: str = "INFO"
    enable_sentry: bool = True
    sentry_environment: str = "staging"

class ProductionConfig(BaseConfig):
    database_url: str
    log_level: str = "WARNING"
    enable_sentry: bool = True
    sentry_environment: str = "production"
    require_https: bool = True

def get_config() -> BaseConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    configs = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.STAGING: StagingConfig,
        Environment.PRODUCTION: ProductionConfig
    }
    
    config_class = configs.get(Environment(env), DevelopmentConfig)
    return config_class()

# Usage
config = get_config()

if config.environment == Environment.PRODUCTION:
    # Production-specific setup
    pass

logger.setLevel(config.log_level)
```

---

## 12. Data Serialization & Validation

### Pydantic Models for Data
**מטרה:** ולידציה ו-typing חזק לכל הדאטה
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

class User(BaseModel):
    id: int
    username: Optional[str] = None
    first_name: str
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
    @validator('username')
    def validate_username(cls, v):
        if v and (len(v) < 3 or len(v) > 32):
            raise ValueError('Username must be 3-32 characters')
        return v

class Message(BaseModel):
    message_id: int
    user_id: int
    text: str = Field(..., max_length=4096)
    timestamp: datetime = Field(default_factory=datetime.now)
    edited: bool = False
    reply_to: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatStatistics(BaseModel):
    total_messages: int = 0
    total_users: int = 0
    active_users_24h: int = 0
    messages_by_hour: List[int] = Field(default_factory=lambda: [0] * 24)
    
    def add_message(self, hour: int):
        self.total_messages += 1
        self.messages_by_hour[hour] += 1

# Usage - automatic validation
try:
    user = User(
        id=123,
        username="ab",  # Too short - will raise ValidationError
        first_name="John"
    )
except ValidationError as e:
    print(e.json())

# Serialize to dict/JSON
user_dict = user.dict()
user_json = user.json()

# Parse from dict
user = User.parse_obj({"id": 123, "first_name": "John"})
```

### Custom Validators
**מטרה:** ולידציות מותאמות אישית לצרכים ספציפיים
```python
from pydantic import BaseModel, validator, root_validator
from typing import Optional
import re

class BotSettings(BaseModel):
    bot_token: str
    webhook_url: Optional[str] = None
    max_message_length: int = 4096
    rate_limit: int = 30
    
    @validator('bot_token')
    def validate_token_format(cls, v):
        """Validate Telegram bot token format"""
        pattern = r'^\d+:[A-Za-z0-9_-]{35}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid bot token format')
        return v
    
    @validator('webhook_url')
    def validate_webhook(cls, v):
        """Ensure webhook uses HTTPS"""
        if v and not v.startswith('https://'):
            raise ValueError('Webhook must use HTTPS')
        return v
    
    @validator('rate_limit')
    def validate_rate_limit(cls, v):
        """Ensure reasonable rate limit"""
        if v < 1 or v > 100:
            raise ValueError('Rate limit must be between 1-100')
        return v
    
    @root_validator
    def validate_settings(cls, values):
        """Cross-field validation"""
        webhook = values.get('webhook_url')
        token = values.get('bot_token')
        
        if webhook:
            # Ensure webhook domain doesn't contain bot token
            if token and token.split(':')[0] in webhook:
                raise ValueError('Webhook URL should not contain bot token')
        
        return values

class UserRegistration(BaseModel):
    email: str
    phone: Optional[str] = None
    age: int
    
    @validator('email')
    def validate_email(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('phone')
    def validate_phone(cls, v):
        if v:
            # Remove common separators
            cleaned = re.sub(r'[-()\s]', '', v)
            if not cleaned.isdigit() or len(cleaned) < 10:
                raise ValueError('Invalid phone number')
            return cleaned
        return v
    
    @validator('age')
    def validate_age(cls, v):
        if v < 13:
            raise ValueError('Must be at least 13 years old')
        if v > 120:
            raise ValueError('Invalid age')
        return v
```

### Serialization Helpers
**מטרה:** המרה נוחה בין פורמטים שונים
```python
from pydantic import BaseModel
from typing import Any, Type, List
import json
from datetime import datetime, date
from decimal import Decimal

class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with support for datetime, Decimal, etc."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)

class SerializationHelper:
    @staticmethod
    def to_json(obj: Any, pretty: bool = False) -> str:
        """Convert object to JSON string"""
        indent = 2 if pretty else None
        return json.dumps(obj, cls=EnhancedJSONEncoder, indent=indent)
    
    @staticmethod
    def from_json(data: str, model: Type[BaseModel]) -> BaseModel:
        """Parse JSON to Pydantic model"""
        return model.parse_raw(data)
    
    @staticmethod
    def batch_serialize(
        items: List[BaseModel],
        exclude_none: bool = True
    ) -> List[dict]:
        """Serialize list of models"""
        return [
            item.dict(exclude_none=exclude_none)
            for item in items
        ]
    
    @staticmethod
    def nested_dict_to_model(
        data: dict,
        model: Type[BaseModel],
        strict: bool = True
    ) -> BaseModel:
        """Convert nested dict to model with error handling"""
        try:
            return model.parse_obj(data)
        except ValidationError as e:
            if strict:
                raise
            # Return partial model with valid fields only
            valid_data = {}
            for field in model.__fields__:
                if field in data:
                    try:
                        valid_data[field] = data[field]
                    except:
                        pass
            return model.parse_obj(valid_data)

# Usage
class UserProfile(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

user = UserProfile(
    id=1,
    name="John",
    email="john@example.com",
    created_at=datetime.now()
)

# Serialize
json_str = SerializationHelper.to_json(user, pretty=True)

# Deserialize
user_restored = SerializationHelper.from_json(json_str, UserProfile)

# Batch operations
users = [user, user2, user3]
users_dict = SerializationHelper.batch_serialize(users)
```

---

**סיכום:** ספרייה זו מכילה את כל הבסיס הנדרש לבניית בוט Telegram מקצועי ומדרגי. כל סניפט ניתן להרחיב ולהתאים לצרכים ספציפיים של הפרויקט שלך.
