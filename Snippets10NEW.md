# Code Snippets - בוט טלגרם ווואטסאפ

סניפטים איכותיים מהריפו של Shipment-bot למפתחים שבונים בוטים וזרימות טלגרם/וואטסאפ.

---

## Placeholder לטלפון בטלגרם

**למה זה שימושי:** טלגרם לא מספק מספר טלפון בהכרח, אבל DB דורש phone_number. יוצר מזהה יציב שלא יפוצץ את הDB.

```python
import hashlib

def _telegram_phone_placeholder(telegram_chat_id: str) -> str:
    """
    יצירת placeholder קצר ל-phone_number עבור משתמשי Telegram.
    מונע כשלי DB כש-phone_number מוגדר NOT NULL.
    """
    if telegram_chat_id is None or str(telegram_chat_id).strip() in ("", "None"):
        raise ValueError("telegram_chat_id is required")

    telegram_chat_id = str(telegram_chat_id).strip()
    candidate = f"tg:{telegram_chat_id}"

    # אם המזהה קצר מספיק - משתמשים בו ישירות
    if len(candidate) <= 20:
        return candidate

    # אחרת - יוצרים hash קצר
    digest = hashlib.sha1(telegram_chat_id.encode("utf-8")).hexdigest()[:17]
    return f"tg:{digest}"
```

---

## חילוץ Chat ID מסוגי עדכונים שונים

**למה זה שימושי:** טלגרם שולח עדכונים בפורמטים שונים (הודעות, callback queries). פונקציה אחת שתמיד מחזירה chat_id נכון.

```python
def _resolve_telegram_chat_id(update: "TelegramUpdate") -> str | None:
    """
    חילוץ chat_id יציב גם עבור callback_query ללא message.
    ב-private chat, user_id == chat_id ולכן אפשר ליפול ל-from_user.id.
    """
    # עדכון רגיל עם הודעה
    if update.message:
        return str(update.message.chat.id)

    # כפתור inline נלחץ
    if update.callback_query:
        cb = update.callback_query
        # קודם מנסים לקבל מההודעה שעליה לחצו
        if cb.message:
            return str(cb.message.chat.id)
        # fallback ל-user ID (שווה ל-chat ID ב-private)
        if cb.from_user:
            return str(cb.from_user.id)

    return None
```

---

## Pydantic עם Field Alias למילים שמורות

**למה זה שימושי:** ה-JSON של טלגרם מכיל שדה `from` שזו מילה שמורה בפייתון. Field alias פותר את זה בצורה אלגנטית.

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class TelegramUser(BaseModel):
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None


class TelegramMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: int
    # 'from' היא מילה שמורה - משתמשים ב-alias
    from_user: Optional[TelegramUser] = Field(default=None, alias="from")
    text: Optional[str] = None
    date: int


class TelegramCallbackQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    from_user: Optional[TelegramUser] = Field(default=None, alias="from")
    data: Optional[str] = None
```

---

## המרת HTML לפורמט וואטסאפ

**למה זה שימושי:** כותבים הודעות ב-HTML אחיד, והפונקציה ממירה אוטומטית לפורמט של וואטסאפ (כוכביות, קווים תחתונים).

```python
import re
import html

def convert_html_to_whatsapp(text: str) -> str:
    """
    ממיר תגי HTML לפורמט וואטסאפ.
    <b> → *, <i> → _, <code> → `
    """
    if not text:
        return ""

    # המרת תגי bold
    result = re.sub(r"<b>(.*?)</b>", r"*\1*", text, flags=re.DOTALL)
    result = re.sub(r"<strong>(.*?)</strong>", r"*\1*", result, flags=re.DOTALL)

    # המרת תגי italic
    result = re.sub(r"<i>(.*?)</i>", r"_\1_", result, flags=re.DOTALL)
    result = re.sub(r"<em>(.*?)</em>", r"_\1_", result, flags=re.DOTALL)

    # המרת תגי strikethrough
    result = re.sub(r"<s>(.*?)</s>", r"~\1~", result, flags=re.DOTALL)
    result = re.sub(r"<del>(.*?)</del>", r"~\1~", result, flags=re.DOTALL)

    # המרת תגי code
    result = re.sub(r"<code>(.*?)</code>", r"`\1`", result, flags=re.DOTALL)
    result = re.sub(r"<pre>(.*?)</pre>", r"```\1```", result, flags=re.DOTALL)

    # המרת <br> לשורה חדשה והסרת תגים לא נתמכים
    result = re.sub(r"<br\s*/?>", "\n", result, flags=re.IGNORECASE)
    result = re.sub(r"<[^>]+>", "", result)

    # המרת HTML entities חזרה לתווים רגילים
    result = html.unescape(result)

    return result
```

---

## פקודות אדמין עם Regex בעברית

**למה זה שימושי:** זיהוי פקודות בעברית מקבוצת מנהלים, עם תמיכה באימוג'י ובפורמטים גמישים ("אשר 123", "✅ אשר שליח 123").

```python
import re
from typing import Optional

async def handle_admin_group_command(db, text: str) -> Optional[str]:
    """
    טיפול בפקודות מנהל מקבוצת הוואטסאפ.
    מזהה פקודות כמו "אשר שליח 123" או "דחה שליח 456"
    """
    text = text.strip()

    # זיהוי פקודת אישור - תומך בפורמטים:
    # "אשר 123", "אשר שליח 123", "✅ אשר 123"
    # ^ מוודא שמתחיל בתחילת ההודעה - מונע התאמה של ציטוטים
    approve_match = re.match(r'^[✅\s]*אשר(?:\s+שליח)?\s+(\d+)\s*$', text)
    if approve_match:
        user_id = int(approve_match.group(1))
        return await _approve_courier(db, user_id)

    # זיהוי פקודת דחייה
    reject_match = re.match(r'^[❌\s]*דחה(?:\s+שליח)?\s+(\d+)\s*$', text)
    if reject_match:
        user_id = int(reject_match.group(1))
        return await _reject_courier(db, user_id)

    return None  # לא זוהתה פקודה
```

---

## Token מאובטח לקישורי Smart Link

**למה זה שימושי:** במקום לחשוף ID של משלוח בקישור (ניתן לנחש), יוצרים token אקראי שלא ניתן לניחוש.

```python
import secrets
from sqlalchemy import Column, String

def generate_secure_token():
    """יצירת token מאובטח URL-safe לקישורי משלוח"""
    return secrets.token_urlsafe(16)


class Delivery(Base):
    __tablename__ = "deliveries"

    id = Column(Integer, primary_key=True)
    # Token מאובטח לקישורים - מונע ניחוש של IDs
    token = Column(
        String(32),
        unique=True,
        nullable=False,
        default=generate_secure_token,
        index=True
    )


# שימוש - תפיסת משלוח לפי token במקום ID
async def capture_delivery_by_token(token: str, courier_id: int):
    """תפיסת משלוח לפי token מאובטח (לא לפי ID)"""
    result = await db.execute(
        select(Delivery).where(Delivery.token == token)
    )
    delivery = result.scalar_one_or_none()

    if not delivery:
        return False, "המשלוח לא נמצא (קישור לא תקין)", None

    # המשך לתפיסה לפי ID הפנימי
    return await capture_delivery(delivery.id, courier_id)
```

---

## תפיסה אטומית עם נעילת שורות (Row Locks)

**למה זה שימושי:** מניעת race conditions - שני שליחים לא יכולים לתפוס אותו משלוח. גם מוודא שיש מספיק קרדיט לפני התפיסה.

```python
from sqlalchemy import select

async def capture_delivery(delivery_id: int, courier_id: int):
    """
    תפיסה אטומית: נעילה → בדיקה → עדכון → commit
    הכל בטרנזקציה אחת.
    """
    try:
        # 1. נעילת רשומת המשלוח (FOR UPDATE)
        delivery_result = await db.execute(
            select(Delivery)
            .where(Delivery.id == delivery_id)
            .with_for_update()  # <- נעילה ברמת שורה
        )
        delivery = delivery_result.scalar_one_or_none()

        if not delivery:
            return False, "המשלוח לא נמצא", None

        # 2. בדיקת סטטוס (אחרי הנעילה!)
        if delivery.status != DeliveryStatus.OPEN:
            return False, "המשלוח כבר נתפס על ידי שליח אחר", None

        # 3. נעילת ארנק השליח
        wallet_result = await db.execute(
            select(CourierWallet)
            .where(CourierWallet.courier_id == courier_id)
            .with_for_update()
        )
        wallet = wallet_result.scalar_one_or_none()

        # 4. בדיקת קרדיט
        fee = delivery.fee
        future_balance = wallet.balance - fee

        if future_balance < wallet.credit_limit:
            return False, f"יתרה לא מספיקה", None

        # 5. עדכון כל הנתונים
        delivery.status = DeliveryStatus.CAPTURED
        delivery.courier_id = courier_id
        wallet.balance = future_balance

        # 6. רישום בledger
        ledger_entry = WalletLedger(
            courier_id=courier_id,
            delivery_id=delivery_id,
            entry_type=LedgerEntryType.DELIVERY_FEE_DEBIT,
            amount=-fee,
            balance_after=future_balance
        )
        db.add(ledger_entry)

        # 7. commit אטומי - הכל או כלום
        await db.commit()

        return True, f"המשלוח נתפס! יתרה: {future_balance}₪", delivery

    except Exception as e:
        await db.rollback()
        raise CaptureError(f"שגיאה בתפיסת המשלוח: {str(e)}")
```

---

## Backoff עם הגנת Overflow

**למה זה שימושי:** חישוב backoff אקספוננציאלי בלי לפוצץ את הזיכרון כשretry_count גדול מדי.

```python
def _calculate_backoff_seconds(
    retry_count: int,
    *,
    base_seconds: int = 60,
    max_backoff_seconds: int = 3600,
) -> int:
    """
    חישוב backoff אקספוננציאלי עם הגנת overflow.
    formula: base * (2 ** retry_count), מוגבל ל-max
    """
    if retry_count < 0:
        retry_count = 0

    if base_seconds <= 0 or max_backoff_seconds <= 0:
        return 0

    # כבר במקסימום?
    if base_seconds >= max_backoff_seconds:
        return max_backoff_seconds

    # חישוב הסף שבו נגיע למקסימום - בלי לחשב 2**retry_count
    required_multiplier = (max_backoff_seconds + base_seconds - 1) // base_seconds
    is_power_of_two = (required_multiplier & (required_multiplier - 1)) == 0
    threshold = required_multiplier.bit_length() - 1
    if not is_power_of_two:
        threshold += 1

    # אם עברנו את הסף - מחזירים מקסימום ישירות
    if retry_count >= threshold:
        return max_backoff_seconds

    # חישוב רגיל (בטוח - לא יגרום ל-overflow)
    backoff = base_seconds * (1 << retry_count)  # 1 << n == 2**n
    return min(backoff, max_backoff_seconds)
```

---

## ניהול Event Loop ב-Celery Tasks

**למה זה שימושי:** Celery רץ sync אבל הקוד שלך async. ה-context manager מוודא ניקוי תקין של resources.

```python
import asyncio
from contextlib import contextmanager

@contextmanager
def get_event_loop():
    """
    Context manager ליצירת event loop נקי ב-Celery.
    מוודא ביטול tasks תלויים וסגירה נקייה.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        try:
            # ביטול כל ה-tasks שעדיין רצים
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # המתנה לביטול
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def run_async(coro):
    """הרצת קוד async ב-Celery task"""
    with get_event_loop() as loop:
        return loop.run_until_complete(coro)


# שימוש ב-Celery task
@celery_app.task
def process_outbox_messages():
    async def _process():
        async with get_task_session() as db:
            # קוד async כאן...
            pass

    return run_async(_process())
```

---

## עדכון JSON Field ב-SQLAlchemy

**למה זה שימושי:** SQLAlchemy לא מזהה שינויים בתוך JSON field. חייבים ליצור dict חדש כדי לגרום ל-dirty flag.

```python
async def update_context(user_id: int, platform: str, key: str, value: any):
    """עדכון שדה בתוך JSON column ב-SQLAlchemy"""
    session = await get_or_create_session(user_id, platform)

    # יצירת copy של ה-dict - חובה!
    # שינוי in-place לא יזוהה על ידי SQLAlchemy
    context = dict(session.context_data or {})  # <- חייבים copy
    context[key] = value
    session.context_data = context  # השמה חדשה מפעילה dirty flag

    await db.commit()


async def transition_to(user_id: int, platform: str, new_state: str, context_update: dict = None):
    """מעבר state עם עדכון context"""
    session = await get_or_create_session(user_id, platform)
    session.current_state = new_state

    if context_update:
        # יצירת dict חדש - לא לשנות in-place!
        current_context = dict(session.context_data or {})
        current_context.update(context_update)
        session.context_data = current_context

    await db.commit()
```

---

## Double-Checked Locking ל-Singleton

**למה זה שימושי:** Circuit breaker צריך להיות singleton, אבל חייבים תמיכה ב-thread safety (Celery workers).

```python
import threading

class CircuitBreaker:
    """Circuit breaker עם singleton pattern thread-safe"""

    _instances: dict[str, "CircuitBreaker"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, service_name: str, config=None) -> "CircuitBreaker":
        """Double-checked locking - מהיר ובטוח"""

        # בדיקה ראשונה - ללא נעילה (fast path)
        if service_name not in cls._instances:
            # נעילה רק אם צריך ליצור
            with cls._instances_lock:
                # בדיקה שנייה - בתוך הנעילה
                # (אולי thread אחר יצר בינתיים)
                if service_name not in cls._instances:
                    cls._instances[service_name] = cls(service_name, config)

        return cls._instances[service_name]


# שימוש - תמיד מקבלים את אותו instance
cb1 = CircuitBreaker.get_instance("telegram")
cb2 = CircuitBreaker.get_instance("telegram")
assert cb1 is cb2  # True
```

---

## מעברי State Machine מותרים

**למה זה שימושי:** הגדרת מעברים חוקיים מונעת באגים - אי אפשר לדלג על שלבים בזרימה.

```python
from enum import Enum

class CourierState(str, Enum):
    INITIAL = "COURIER.INITIAL"
    AWAITING_NAME = "COURIER.AWAITING_NAME"
    AWAITING_AREA = "COURIER.AWAITING_AREA"
    AWAITING_DOCUMENT = "COURIER.AWAITING_DOCUMENT"
    PENDING_APPROVAL = "COURIER.PENDING_APPROVAL"
    MENU = "COURIER.MENU"


# הגדרת מעברים מותרים - מכל state לאן אפשר להגיע
COURIER_TRANSITIONS = {
    CourierState.INITIAL: {CourierState.AWAITING_NAME},
    CourierState.AWAITING_NAME: {CourierState.AWAITING_AREA, CourierState.INITIAL},
    CourierState.AWAITING_AREA: {CourierState.AWAITING_DOCUMENT, CourierState.AWAITING_NAME},
    CourierState.AWAITING_DOCUMENT: {CourierState.PENDING_APPROVAL, CourierState.AWAITING_AREA},
    CourierState.PENDING_APPROVAL: {CourierState.MENU},
    CourierState.MENU: {CourierState.INITIAL},  # איפוס
}


def _is_valid_transition(current: str, target: str) -> bool:
    """בדיקה האם מעבר מותר"""
    try:
        current_state = CourierState(current)
        target_state = CourierState(target)

        if current_state in COURIER_TRANSITIONS:
            return target_state in COURIER_TRANSITIONS[current_state]
    except ValueError:
        pass

    return False


async def transition_to(user_id: int, new_state: str) -> bool:
    """מעבר רק אם מותר"""
    current = await get_current_state(user_id)

    if not _is_valid_transition(current, new_state):
        logger.warning(
            "Invalid transition",
            extra_data={"current": current, "target": new_state}
        )
        return False

    # ביצוע המעבר...
    return True
```

---

## Transactional Outbox Pattern

**למה זה שימושי:** שומרים הודעות בDB באותה טרנזקציה עם הלוגיקה העסקית. Worker נפרד שולח - מבטיח שלא נאבד הודעות.

```python
from enum import Enum

class MessageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"


class OutboxMessage(Base):
    """הודעות ממתינות לשליחה"""
    __tablename__ = "outbox_messages"

    id = Column(Integer, primary_key=True)
    platform = Column(String(20))  # "whatsapp" / "telegram"
    recipient_id = Column(String(100))  # מספר טלפון או chat_id
    message_type = Column(String(50))
    message_content = Column(JSON)
    status = Column(Enum(MessageStatus), default=MessageStatus.PENDING)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=5)
    next_retry_at = Column(DateTime, nullable=True)


async def queue_delivery_broadcast(delivery: Delivery):
    """הוספת הודעה ל-outbox באותה טרנזקציה"""
    content = {
        "delivery_id": delivery.id,
        "token": delivery.token,
        "message_text": f"🚚 משלוח חדש! /capture {delivery.token}"
    }

    # BROADCAST_COURIERS = הworker יפרוש לכל השליחים
    message = OutboxMessage(
        platform="telegram",
        recipient_id="BROADCAST_COURIERS",
        message_type="delivery_broadcast",
        message_content=content,
        status=MessageStatus.PENDING
    )
    db.add(message)
    # לא commit - יקרה ביחד עם יצירת המשלוח!
```

---

## הבחנה בין קבוצות להודעות פרטיות

**למה זה שימושי:** בוואטסאפ צריך להבחין בין הודעות מקבוצות (admin group) להודעות פרטיות מלקוחות.

```python
async def whatsapp_webhook(payload: WhatsAppWebhookPayload, db):
    for message in payload.messages:
        sender_id = message.sender_id or message.from_number

        # בדיקה אם ההודעה מגיעה מקבוצה
        is_group_message = sender_id.endswith("@g.us")

        if is_group_message:
            # בדיקה אם זו קבוצת המנהלים
            if sender_id == settings.WHATSAPP_ADMIN_GROUP_ID:
                logger.info("Admin group message", extra_data={"text": message.text[:50]})

                # ניסיון לזהות פקודת מנהל
                response = await handle_admin_group_command(db, message.text)

                if response:
                    # שליחת תגובה לקבוצה
                    await send_whatsapp_message(sender_id, response)
                # הודעות לא-פקודות בקבוצה - מתעלמים
            else:
                # הודעה מקבוצה אחרת - מתעלמים
                logger.debug("Non-admin group, ignoring")

            continue  # לא ממשיכים לזרימה הרגילה

        # הודעה פרטית - המשך טיפול רגיל...
        user, is_new = await get_or_create_user(db, sender_id)
```

---

## שליחה מקבילית עם סיכום תוצאות

**למה זה שימושי:** שליחת broadcast לאלפי משתמשים - מקבילית ומהירה, עם ספירת הצלחות/כשלונות.

```python
import asyncio

async def broadcast_to_couriers(message_text: str):
    """שליחה מקבילית לכל השליחים עם סיכום"""

    # שליפת כל השליחים הפעילים
    whatsapp_couriers = await get_active_couriers("whatsapp")
    telegram_couriers = await get_active_couriers("telegram")

    content = {"message_text": message_text}

    # יצירת כל ה-tasks
    tasks = []
    for courier in whatsapp_couriers:
        tasks.append(_send_whatsapp_message(courier.phone_number, content))
    for courier in telegram_couriers:
        if courier.telegram_chat_id:  # סינון חסרי chat_id
            tasks.append(_send_telegram_message(courier.telegram_chat_id, content))

    if not tasks:
        return {"error": "No active couriers", "total_sent": 0}

    # הרצה מקבילית
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # סיכום תוצאות
    final_results = []
    for r in results:
        if isinstance(r, Exception):
            final_results.append({"success": False, "error": str(r)})
        else:
            final_results.append({"success": r})

    successful = sum(1 for r in final_results if r.get("success"))

    return {
        "total_sent": len(final_results),
        "successful": successful,
        "failed": len(final_results) - successful,
        "results": final_results
    }
```

---

## סטטוס "דביק" שלא ניתן לשנות

**למה זה שימושי:** משתמש BLOCKED לא יכול להפוך למאושר - הגנה מפני שגיאות אדמין.

```python
async def _approve_courier(db, user_id: int) -> str:
    """אישור שליח עם בדיקת סטטוסים דביקים"""
    user = await get_user(db, user_id)

    if not user:
        return f"❌ לא נמצא משתמש {user_id}"

    if user.role != UserRole.COURIER:
        return f"❌ משתמש {user_id} אינו שליח"

    if user.approval_status == ApprovalStatus.APPROVED:
        return f"ℹ️ שליח {user_id} כבר מאושר"

    # סטטוס BLOCKED הוא "דביק" - לא ניתן לשנות
    if user.approval_status == ApprovalStatus.BLOCKED:
        return f"⛔ שליח {user_id} חסום. לא ניתן לאשר משתמש חסום."

    # אישור
    user.approval_status = ApprovalStatus.APPROVED
    await db.commit()

    logger.info("Courier approved", extra_data={"user_id": user_id})

    # שליחת הודעה לשליח
    await notify_user_approved(user)

    return f"✅ שליח {user_id} אושר בהצלחה!"
```

---

## מניעת התראות כפולות במעבר State

**למה זה שימושי:** שמירת state קודם לפני טיפול - שולחים התראה רק במעבר הראשון ל-state חדש.

```python
async def handle_courier_message(user, text, db):
    state_manager = StateManager(db)

    # שמירת המצב הקודם לפני הטיפול
    previous_state = await state_manager.get_current_state(user.id, "telegram")

    # טיפול בהודעה - עלול לשנות state
    handler = CourierStateHandler(db)
    response, new_state = await handler.handle_message(user, text)

    # שליחת התראה רק במעבר הראשון ל-PENDING_APPROVAL
    # (לא אם המשתמש כבר היה ב-state הזה)
    if (new_state == CourierState.PENDING_APPROVAL.value and
        previous_state != CourierState.PENDING_APPROVAL.value and
        user.approval_status == ApprovalStatus.PENDING):

        # שליחת התראה למנהלים - פעם אחת בלבד
        await notify_admins_new_registration(user)

    return response, new_state
```

---

## כפתורי Inline עם Callback Data

**למה זה שימושי:** יצירת מקלדת inline לטלגרם - כפתורים שנשארים על ההודעה ושולחים callback_data.

```python
async def send_telegram_message(chat_id: str, text: str, keyboard: list = None, inline: bool = False):
    """שליחת הודעה עם תמיכה בשני סוגי מקלדות"""

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }

    if keyboard:
        if inline:
            # מקלדת inline - כפתורים על ההודעה
            inline_keyboard = []
            for row in keyboard:
                inline_row = []
                for button_text in row:
                    inline_row.append({
                        "text": button_text,
                        "callback_data": button_text  # הערך שיחזור ב-callback_query
                    })
                inline_keyboard.append(inline_row)

            payload["reply_markup"] = {
                "inline_keyboard": inline_keyboard
            }
        else:
            # מקלדת רגילה - מחליפה את המקלדת
            payload["reply_markup"] = {
                "keyboard": keyboard,
                "resize_keyboard": True,      # התאמה לגודל
                "one_time_keyboard": True     # נעלמת אחרי לחיצה
            }

    await send_to_telegram_api(payload)
```

---

## Force State ללא ולידציה (איפוס)

**למה זה שימושי:** לפעמים צריך לאפס את המשתמש לתחילת הזרימה, בלי להתחשב במעברים המותרים.

```python
async def force_state(user_id: int, platform: str, new_state: str, context: dict = None):
    """
    כפיית state ללא ולידציה - לשימוש באיפוסים ופעולות אדמין.
    לא בודק מעברים מותרים!
    """
    session = await get_or_create_session(user_id, platform)
    session.current_state = new_state

    if context is not None:
        session.context_data = context  # החלפה מלאה של ה-context

    await db.commit()


# שימוש - טיפול בפקודת /start
if text.strip().startswith("/start"):
    # איפוס מלא - ללא קשר ל-state הנוכחי
    await state_manager.force_state(
        user.id,
        "telegram",
        CourierState.MENU.value,
        context={}  # ניקוי context
    )
```

---
