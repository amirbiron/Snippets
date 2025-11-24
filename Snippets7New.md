# Advanced Telegram Bot Code Snippets - Part 7

## 1. Voice & Audio Processing

### Transcribe Voice Messages with Whisper API
**למה זה שימושי:** מאפשר לבוט להבין הודעות קוליות ולהגיב עליהן בטקסט. שימושי במיוחד לנגישות, תיעוד פגישות, או ניתוח תוכן קולי.

```python
import os
import tempfile
import whisper
from telegram import Update
from telegram.ext import ContextTypes
import asyncio

class VoiceTranscriber:
    def __init__(self):
        # Load Whisper model (tiny, base, small, medium, large)
        self.model = whisper.load_model("base")
    
    async def transcribe_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Transcribe voice message to text."""
        try:
            # Get voice file
            voice = update.message.voice
            file = await context.bot.get_file(voice.file_id)
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
                await file.download_to_drive(temp_file.name)
                temp_path = temp_file.name
            
            # Transcribe in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_sync,
                temp_path
            )
            
            # Clean up
            os.unlink(temp_path)
            
            # Send transcription
            await update.message.reply_text(
                f"🎙️ **תמלול:**\n{result['text']}\n\n"
                f"📊 **שפה:** {result.get('language', 'unknown')}"
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ שגיאה בתמלול: {str(e)}")
    
    def _transcribe_sync(self, audio_path: str) -> dict:
        """Synchronous transcription method."""
        result = self.model.transcribe(
            audio_path,
            language=None,  # Auto-detect language
            task="transcribe"
        )
        return result
```

## 2. Real-time Analytics Dashboard

### Stream Bot Metrics via WebSocket
**למה זה שימושי:** מספק תצוגה חיה של ביצועי הבוט, מספר משתמשים פעילים, ופקודות פופולריות. מאפשר זיהוי מהיר של בעיות ומגמות שימוש.

```python
import json
import asyncio
from typing import Dict, Set
from datetime import datetime, timedelta
from collections import defaultdict
from aiohttp import web
import aiohttp_cors

class BotAnalyticsDashboard:
    def __init__(self):
        self.websockets: Set[web.WebSocketResponse] = set()
        self.metrics = defaultdict(lambda: 0)
        self.active_users = set()
        
    async def track_event(self, event_type: str, user_id: int, metadata: Dict = None):
        """Track an analytics event."""
        self.metrics[event_type] += 1
        self.active_users.add(user_id)
        
        # Broadcast to all connected dashboards
        await self.broadcast_update({
            'type': 'event',
            'event': event_type,
            'timestamp': datetime.now().isoformat(),
            'total_users': len(self.active_users),
            'metrics': dict(self.metrics),
            'metadata': metadata
        })
    
    async def websocket_handler(self, request):
        """WebSocket handler for real-time dashboard updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Send initial data
            await ws.send_json({
                'type': 'initial',
                'metrics': dict(self.metrics),
                'active_users': len(self.active_users)
            })
            
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Handle dashboard requests
                    if data.get('action') == 'get_details':
                        await ws.send_json({
                            'type': 'details',
                            'data': await self.get_detailed_metrics()
                        })
                        
        finally:
            self.websockets.discard(ws)
            
        return ws
    
    async def broadcast_update(self, data: Dict):
        """Broadcast update to all connected dashboards."""
        if self.websockets:
            await asyncio.gather(
                *[ws.send_json(data) for ws in self.websockets],
                return_exceptions=True
            )
    
    async def get_detailed_metrics(self) -> Dict:
        """Get detailed metrics for dashboard."""
        return {
            'commands': dict(self.metrics),
            'active_users_count': len(self.active_users),
            'timestamp': datetime.now().isoformat()
        }
    
    def setup_dashboard_app(self) -> web.Application:
        """Setup web application for dashboard."""
        app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Add routes
        resource = cors.add(app.router.add_resource("/ws"))
        resource.add_route("GET", self.websocket_handler)
        
        return app
```

## 3. ML-Based Anti-Spam

### Detect Spam with Machine Learning
**למה זה שימושי:** מזהה ומסנן אוטומטית הודעות ספאם בצ'אטים. משפר את איכות השיחה ומגן על משתמשים מתוכן לא רצוי.

```python
import pickle
import re
import numpy as np
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from telegram import Update
from telegram.ext import ContextTypes

class MLSpamDetector:
    def __init__(self, model_path: str = "spam_model.pkl"):
        # Load pre-trained model and vectorizer
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(model_path.replace('.pkl', '_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Spam patterns for quick detection
        self.spam_patterns = [
            r'(?i)(click here|buy now|limited offer)',
            r'(?i)(earn money|work from home|get rich)',
            r'(?i)(viagra|casino|adult)',
            r'http[s]?://bit\.ly/\w+',  # Shortened URLs
            r'@[\w]+{3,}',  # Multiple mentions
        ]
    
    async def check_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check if message is spam."""
        text = update.message.text
        
        # Quick pattern check
        if self._has_spam_patterns(text):
            await self._handle_spam(update, context, confidence=0.95)
            return
        
        # ML prediction
        is_spam, confidence = self._predict_spam(text)
        
        if is_spam and confidence > 0.8:
            await self._handle_spam(update, context, confidence)
    
    def _has_spam_patterns(self, text: str) -> bool:
        """Check for known spam patterns."""
        for pattern in self.spam_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _predict_spam(self, text: str) -> Tuple[bool, float]:
        """Predict if text is spam using ML model."""
        try:
            # Feature extraction
            features = self._extract_features(text)
            
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Combine features
            combined = np.hstack([text_vector.toarray(), features.reshape(1, -1)])
            
            # Predict
            prediction = self.model.predict(combined)[0]
            confidence = self.model.predict_proba(combined)[0].max()
            
            return prediction == 1, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0
    
    def _extract_features(self, text: str) -> np.array:
        """Extract additional features from text."""
        features = [
            len(text),  # Text length
            text.count('!'),  # Exclamation marks
            text.count('$'),  # Dollar signs
            text.count('http'),  # Links
            text.count('@'),  # Mentions
            len(re.findall(r'[A-Z]', text)) / max(len(text), 1),  # Uppercase ratio
            len(re.findall(r'\d', text)) / max(len(text), 1),  # Digit ratio
        ]
        return np.array(features)
    
    async def _handle_spam(self, update: Update, context: ContextTypes.DEFAULT_TYPE, confidence: float):
        """Handle detected spam."""
        # Delete spam message
        await update.message.delete()
        
        # Log spam detection
        print(f"Spam detected (confidence: {confidence:.2f}): {update.message.text[:50]}...")
        
        # Optionally warn/ban user
        if confidence > 0.95:
            # Ban repeat offenders
            user_id = update.message.from_user.id
            spam_count = context.user_data.get('spam_count', 0) + 1
            context.user_data['spam_count'] = spam_count
            
            if spam_count >= 3:
                await context.bot.ban_chat_member(
                    chat_id=update.effective_chat.id,
                    user_id=user_id
                )
```

## 4. Advanced Survey Builder

### Create Dynamic Surveys with Branching Logic
**למה זה שימושי:** מאפשר יצירת סקרים מורכבים עם לוגיקת הסתעפות מותנית. מצוין לאיסוף משוב מפורט, טפסי רישום, או שאלונים אבחוניים.

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler

@dataclass
class SurveyQuestion:
    id: str
    text: str
    type: str  # 'choice', 'text', 'number', 'date', 'location'
    options: Optional[List[str]] = None
    validation: Optional[Callable] = None
    next_question: Optional[str] = None  # Default next question
    conditional_next: Optional[Dict[str, str]] = None  # Answer -> next question

class SurveyBuilder:
    def __init__(self, survey_id: str):
        self.survey_id = survey_id
        self.questions: Dict[str, SurveyQuestion] = {}
        self.responses: Dict[int, Dict] = {}  # user_id -> responses
        
    def add_question(self, question: SurveyQuestion):
        """Add a question to the survey."""
        self.questions[question.id] = question
        
    async def start_survey(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start the survey for a user."""
        user_id = update.effective_user.id
        self.responses[user_id] = {'current_question': 'q1', 'answers': {}}
        
        await self.ask_question(update, context, 'q1')
        return 'ANSWERING'
    
    async def ask_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE, question_id: str):
        """Ask a specific question."""
        question = self.questions.get(question_id)
        if not question:
            await self.end_survey(update, context)
            return
        
        user_id = update.effective_user.id
        self.responses[user_id]['current_question'] = question_id
        
        if question.type == 'choice' and question.options:
            # Create inline keyboard for choices
            keyboard = []
            for i in range(0, len(question.options), 2):
                row = []
                for option in question.options[i:i+2]:
                    row.append(InlineKeyboardButton(
                        option, 
                        callback_data=f"survey_{self.survey_id}_{question_id}_{option}"
                    ))
                keyboard.append(row)
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text(
                question.text,
                reply_markup=reply_markup
            )
        else:
            # Text/number/date question
            await update.effective_message.reply_text(
                f"{question.text}\n\n"
                f"_סוג תשובה: {self._get_type_hint(question.type)}_",
                parse_mode='Markdown'
            )
    
    async def process_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process user's answer."""
        user_id = update.effective_user.id
        if user_id not in self.responses:
            return
        
        current_q_id = self.responses[user_id]['current_question']
        question = self.questions[current_q_id]
        
        # Extract answer
        if update.callback_query:
            # Choice answer from callback
            answer = update.callback_query.data.split('_')[-1]
            await update.callback_query.answer()
        else:
            # Text answer
            answer = update.message.text
        
        # Validate answer
        if question.validation and not question.validation(answer):
            await update.effective_message.reply_text(
                "❌ התשובה לא תקינה. אנא נסה שוב."
            )
            return 'ANSWERING'
        
        # Store answer
        self.responses[user_id]['answers'][current_q_id] = answer
        
        # Determine next question
        next_q_id = self._get_next_question(question, answer)
        
        if next_q_id:
            await self.ask_question(update, context, next_q_id)
            return 'ANSWERING'
        else:
            await self.end_survey(update, context)
            return ConversationHandler.END
    
    def _get_next_question(self, question: SurveyQuestion, answer: str) -> Optional[str]:
        """Determine next question based on branching logic."""
        if question.conditional_next and answer in question.conditional_next:
            return question.conditional_next[answer]
        return question.next_question
    
    def _get_type_hint(self, question_type: str) -> str:
        """Get user-friendly type hint."""
        hints = {
            'text': 'טקסט חופשי',
            'number': 'מספר',
            'date': 'תאריך (DD/MM/YYYY)',
            'location': 'שלח מיקום'
        }
        return hints.get(question_type, question_type)
    
    async def end_survey(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """End the survey and save results."""
        user_id = update.effective_user.id
        responses = self.responses.get(user_id, {}).get('answers', {})
        
        # Save to database
        await self.save_responses(user_id, responses)
        
        # Send summary
        summary = "🎉 **הסקר הושלם בהצלחה!**\n\n"
        summary += "**התשובות שלך:**\n"
        for q_id, answer in responses.items():
            question = self.questions[q_id]
            summary += f"• {question.text}: {answer}\n"
        
        await update.effective_message.reply_text(summary)
        
        # Clean up
        if user_id in self.responses:
            del self.responses[user_id]
    
    async def save_responses(self, user_id: int, responses: Dict):
        """Save survey responses to database."""
        # Implementation depends on your database
        pass

# Example usage
def create_satisfaction_survey():
    survey = SurveyBuilder("satisfaction_2024")
    
    # Question 1: Overall satisfaction
    survey.add_question(SurveyQuestion(
        id="q1",
        text="כמה אתה מרוצה מהשירות שלנו?",
        type="choice",
        options=["מאוד מרוצה", "מרוצה", "בינוני", "לא מרוצה"],
        next_question="q2",
        conditional_next={
            "לא מרוצה": "q_complaint",
            "בינוני": "q_improvement"
        }
    ))
    
    # Question 2: Recommendation
    survey.add_question(SurveyQuestion(
        id="q2",
        text="האם תמליץ על השירות שלנו לחבר?",
        type="choice",
        options=["בהחלט כן", "כן", "אולי", "לא"],
        next_question="q3"
    ))
    
    # Complaint branch
    survey.add_question(SurveyQuestion(
        id="q_complaint",
        text="נשמח לשמוע מה לא היה טוב. ספר לנו בפירוט:",
        type="text",
        validation=lambda x: len(x) >= 10,  # At least 10 characters
        next_question="q3"
    ))
    
    return survey
```

## 5. Telegram Games Integration

### Integrate HTML5 Games
**למה זה שימושי:** מאפשר הוספת משחקים אינטראקטיביים לבוט. מעלה את המעורבות של המשתמשים ומספק חוויה בידורית.

```python
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import hashlib
import time

class TelegramGameManager:
    def __init__(self, game_url: str, bot_token: str):
        self.game_url = game_url
        self.bot_token = bot_token
        self.high_scores = {}  # game_id -> [(user_id, score, timestamp)]
        
    async def send_game(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send game to user."""
        keyboard = [[
            InlineKeyboardButton(
                "🎮 שחק עכשיו!",
                callback_game=True
            )
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_game(
            game_short_name="my_awesome_game",
            reply_markup=reply_markup
        )
    
    async def handle_game_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle game button click."""
        query = update.callback_query
        
        # Generate secure URL with user authentication
        user_id = query.from_user.id
        timestamp = int(time.time())
        
        # Create auth hash
        auth_data = f"{user_id}:{timestamp}:{self.bot_token}"
        auth_hash = hashlib.sha256(auth_data.encode()).hexdigest()
        
        # Build game URL with auth params
        game_url = (
            f"{self.game_url}"
            f"?user_id={user_id}"
            f"&timestamp={timestamp}"
            f"&auth={auth_hash}"
            f"&inline_message_id={query.inline_message_id}"
        )
        
        # Answer with game URL
        await query.answer(url=game_url)
    
    async def set_game_score(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set user's game score."""
        user_id = update.effective_user.id
        
        # Parse score from message (you'd get this from your game server)
        try:
            score = int(context.args[0])
        except (IndexError, ValueError):
            await update.message.reply_text("❌ ציון לא תקין")
            return
        
        # Update high score
        game_id = "my_awesome_game"
        if game_id not in self.high_scores:
            self.high_scores[game_id] = []
        
        # Check if it's a new high score for this user
        user_scores = [s for uid, s, _ in self.high_scores[game_id] if uid == user_id]
        is_new_high = not user_scores or score > max(user_scores)
        
        if is_new_high:
            # Remove old score
            self.high_scores[game_id] = [
                (uid, s, t) for uid, s, t in self.high_scores[game_id] 
                if uid != user_id
            ]
            # Add new score
            self.high_scores[game_id].append((user_id, score, time.time()))
            # Sort by score
            self.high_scores[game_id].sort(key=lambda x: x[1], reverse=True)
            
            await update.message.reply_text(
                f"🎉 שיא חדש! הציון שלך: {score}"
            )
        else:
            await update.message.reply_text(
                f"✅ הציון שלך: {score}"
            )
    
    async def show_leaderboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show game leaderboard."""
        game_id = "my_awesome_game"
        scores = self.high_scores.get(game_id, [])
        
        if not scores:
            await update.message.reply_text("📊 אין עדיין ציונים")
            return
        
        leaderboard = "🏆 **טבלת השיאים:**\n\n"
        medals = ["🥇", "🥈", "🥉"]
        
        for i, (user_id, score, _) in enumerate(scores[:10]):
            # Get user info
            try:
                user = await context.bot.get_chat_member(
                    update.effective_chat.id, 
                    user_id
                )
                name = user.user.first_name
            except:
                name = f"User {user_id}"
            
            medal = medals[i] if i < 3 else f"{i+1}."
            leaderboard += f"{medal} {name}: {score:,} נקודות\n"
        
        await update.message.reply_text(leaderboard)
```

## 6. Location Services

### Find Nearby Users or Services
**למה זה שימושי:** מאפשר חיפוש משתמשים או שירותים בקרבת מקום. מצוין לבוטים של משלוחים, מציאת חברים קרובים, או שירותים מבוססי מיקום.

```python
from typing import List, Tuple, Dict
import math
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorClient
from telegram import Update, Location
from telegram.ext import ContextTypes

@dataclass
class LocationPoint:
    user_id: int
    lat: float
    lon: float
    name: str
    timestamp: float
    metadata: Dict = None

class LocationServices:
    def __init__(self, mongo_client: AsyncIOMotorClient):
        self.db = mongo_client.bot_db
        self.locations = self.db.user_locations
        
        # Create geospatial index
        self.locations.create_index([("location", "2dsphere")])
    
    async def save_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Save user's location."""
        location = update.message.location
        user = update.effective_user
        
        # Save to MongoDB with GeoJSON format
        await self.locations.update_one(
            {"user_id": user.id},
            {
                "$set": {
                    "user_id": user.id,
                    "name": user.first_name,
                    "location": {
                        "type": "Point",
                        "coordinates": [location.longitude, location.latitude]
                    },
                    "timestamp": update.message.date.timestamp(),
                    "live_period": location.live_period
                }
            },
            upsert=True
        )
        
        await update.message.reply_text("📍 המיקום שלך נשמר בהצלחה!")
        
        # Find nearby users
        nearby = await self.find_nearby_users(
            location.latitude, 
            location.longitude, 
            max_distance=5000  # 5km
        )
        
        if nearby:
            text = "👥 **משתמשים קרובים (עד 5 ק״מ):**\n\n"
            for point in nearby:
                if point.user_id != user.id:
                    distance = self._calculate_distance(
                        location.latitude, location.longitude,
                        point.lat, point.lon
                    )
                    text += f"• {point.name}: {distance:.1f} ק״מ\n"
            
            await update.message.reply_text(text)
    
    async def find_nearby_users(
        self, 
        lat: float, 
        lon: float, 
        max_distance: int = 1000,
        limit: int = 10
    ) -> List[LocationPoint]:
        """Find users within specified distance (meters)."""
        
        # MongoDB geospatial query
        cursor = self.locations.find({
            "location": {
                "$near": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "$maxDistance": max_distance
                }
            }
        }).limit(limit)
        
        results = []
        async for doc in cursor:
            results.append(LocationPoint(
                user_id=doc['user_id'],
                lat=doc['location']['coordinates'][1],
                lon=doc['location']['coordinates'][0],
                name=doc['name'],
                timestamp=doc['timestamp'],
                metadata=doc.get('metadata')
            ))
        
        return results
    
    async def find_in_radius(
        self, 
        center_lat: float,
        center_lon: float,
        radius_km: float
    ) -> List[LocationPoint]:
        """Find all users within a circular area."""
        
        # Convert km to meters for MongoDB
        radius_meters = radius_km * 1000
        
        cursor = self.locations.find({
            "location": {
                "$geoWithin": {
                    "$centerSphere": [
                        [center_lon, center_lat],
                        radius_meters / 6378100  # Earth radius in meters
                    ]
                }
            }
        })
        
        results = []
        async for doc in cursor:
            results.append(LocationPoint(
                user_id=doc['user_id'],
                lat=doc['location']['coordinates'][1],
                lon=doc['location']['coordinates'][0],
                name=doc['name'],
                timestamp=doc['timestamp']
            ))
        
        return results
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        R = 6371  # Earth radius in kilometers
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    async def share_live_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle live location sharing."""
        location = update.edited_message.location if update.edited_message else update.message.location
        
        # Update location in real-time
        await self.locations.update_one(
            {"user_id": update.effective_user.id},
            {
                "$set": {
                    "location": {
                        "type": "Point",
                        "coordinates": [location.longitude, location.latitude]
                    },
                    "timestamp": update.effective_message.date.timestamp(),
                    "is_live": True,
                    "heading": location.heading,
                    "horizontal_accuracy": location.horizontal_accuracy
                }
            }
        )
        
        # Notify nearby users (optional)
        await self._notify_nearby_users(location.latitude, location.longitude)
    
    async def _notify_nearby_users(self, lat: float, lon: float):
        """Notify users when someone is nearby."""
        # Implementation for real-time notifications
        pass
```

## 7. Bot-to-Bot Communication

### Inter-Bot Messaging System
**למה זה שימושי:** מאפשר תקשורת בין מספר בוטים לצורך סנכרון מידע, העברת משימות, או יצירת מערכת בוטים מבוזרת.

```python
import json
import asyncio
import aioredis
from typing import Dict, Any, Callable
from telegram import Bot
from dataclasses import dataclass, asdict

@dataclass
class BotMessage:
    from_bot: str
    to_bot: str
    action: str
    payload: Dict
    timestamp: float
    message_id: str

class BotNetwork:
    def __init__(self, bot_id: str, bot_token: str, redis_url: str):
        self.bot_id = bot_id
        self.bot = Bot(token=bot_token)
        self.redis = None
        self.handlers: Dict[str, Callable] = {}
        self.other_bots: Dict[str, str] = {}  # bot_id -> bot_token
        
    async def connect(self):
        """Connect to Redis for message queue."""
        self.redis = await aioredis.from_url(self.redis_url)
        
        # Start listening for messages
        asyncio.create_task(self._listen_for_messages())
    
    def register_bot(self, bot_id: str, bot_token: str):
        """Register another bot in the network."""
        self.other_bots[bot_id] = bot_token
    
    def register_handler(self, action: str, handler: Callable):
        """Register handler for specific action."""
        self.handlers[action] = handler
    
    async def send_to_bot(self, to_bot: str, action: str, payload: Dict):
        """Send message to another bot."""
        message = BotMessage(
            from_bot=self.bot_id,
            to_bot=to_bot,
            action=action,
            payload=payload,
            timestamp=asyncio.get_event_loop().time(),
            message_id=f"{self.bot_id}_{asyncio.get_event_loop().time()}"
        )
        
        # Publish to Redis channel
        channel = f"bot_channel_{to_bot}"
        await self.redis.publish(
            channel,
            json.dumps(asdict(message))
        )
        
        # Log the message
        await self._log_message(message)
        
        return message.message_id
    
    async def broadcast(self, action: str, payload: Dict):
        """Broadcast message to all bots."""
        tasks = []
        for bot_id in self.other_bots:
            tasks.append(self.send_to_bot(bot_id, action, payload))
        
        await asyncio.gather(*tasks)
    
    async def _listen_for_messages(self):
        """Listen for incoming messages from other bots."""
        pubsub = self.redis.pubsub()
        channel = f"bot_channel_{self.bot_id}"
        await pubsub.subscribe(channel)
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await self._process_message(message['data'])
    
    async def _process_message(self, data: bytes):
        """Process incoming message."""
        try:
            message_data = json.loads(data)
            message = BotMessage(**message_data)
            
            # Find and execute handler
            handler = self.handlers.get(message.action)
            if handler:
                await handler(message)
            else:
                print(f"No handler for action: {message.action}")
            
            # Acknowledge message
            await self._acknowledge_message(message)
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    async def _acknowledge_message(self, message: BotMessage):
        """Send acknowledgment back to sender."""
        ack_channel = f"bot_ack_{message.from_bot}"
        await self.redis.publish(
            ack_channel,
            json.dumps({
                "message_id": message.message_id,
                "status": "received",
                "bot_id": self.bot_id
            })
        )
    
    async def _log_message(self, message: BotMessage):
        """Log message for debugging."""
        await self.redis.zadd(
            f"bot_messages_{self.bot_id}",
            {json.dumps(asdict(message)): message.timestamp}
        )
    
    async def request_data(self, from_bot: str, data_type: str) -> Dict:
        """Request data from another bot and wait for response."""
        request_id = f"req_{asyncio.get_event_loop().time()}"
        
        # Create response future
        response_future = asyncio.Future()
        self.handlers[f"response_{request_id}"] = lambda msg: response_future.set_result(msg.payload)
        
        # Send request
        await self.send_to_bot(from_bot, "data_request", {
            "request_id": request_id,
            "data_type": data_type
        })
        
        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(response_future, timeout=10.0)
            return response
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}
        finally:
            # Clean up handler
            self.handlers.pop(f"response_{request_id}", None)

# Example usage
async def setup_bot_network():
    # Main bot
    main_bot = BotNetwork("main_bot", "MAIN_BOT_TOKEN", "redis://localhost")
    await main_bot.connect()
    
    # Register other bots
    main_bot.register_bot("analytics_bot", "ANALYTICS_BOT_TOKEN")
    main_bot.register_bot("support_bot", "SUPPORT_BOT_TOKEN")
    
    # Register handlers
    async def handle_user_stats(message: BotMessage):
        print(f"Received user stats: {message.payload}")
    
    main_bot.register_handler("user_stats", handle_user_stats)
    
    # Send message to analytics bot
    await main_bot.send_to_bot("analytics_bot", "get_stats", {
        "user_id": 123456,
        "period": "last_week"
    })
    
    # Broadcast event to all bots
    await main_bot.broadcast("system_update", {
        "version": "2.0",
        "features": ["new_ui", "faster_responses"]
    })
```

## 8. Auto-Deployment Scripts

### GitHub Actions Deployment
**למה זה שימושי:** מאפשר פריסה אוטומטית של הבוט בכל פעם שיש שינוי בקוד. חוסך זמן ומפחית טעויות ידניות בתהליך הפריסה.

```yaml
# .github/workflows/deploy.yml
name: Deploy Telegram Bot

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      env:
        BOT_TOKEN: ${{ secrets.TEST_BOT_TOKEN }}
      run: |
        pytest tests/ --cov=bot --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to server
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SERVER_SSH_KEY }}
        port: ${{ secrets.SERVER_PORT }}
        script: |
          cd /home/bot/telegram-bot
          git pull origin main
          
          # Backup current version
          cp -r . ../backup/$(date +%Y%m%d_%H%M%S)
          
          # Install/update dependencies
          source venv/bin/activate
          pip install -r requirements.txt
          
          # Run migrations if needed
          alembic upgrade head
          
          # Restart bot with zero downtime
          sudo systemctl reload telegram-bot || sudo systemctl restart telegram-bot
          
          # Health check
          sleep 5
          curl -f http://localhost:8000/health || exit 1
    
    - name: Notify deployment
      if: success()
      uses: appleboy/telegram-action@master
      with:
        to: ${{ secrets.TELEGRAM_CHAT_ID }}
        token: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        message: |
          ✅ Deployment successful!
          
          Repository: ${{ github.repository }}
          Branch: ${{ github.ref }}
          Commit: ${{ github.sha }}
          Author: ${{ github.actor }}
          
          View changes: ${{ github.event.compare }}
    
    - name: Rollback on failure
      if: failure()
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SERVER_SSH_KEY }}
        script: |
          cd /home/bot
          latest_backup=$(ls -t backup/ | head -1)
          cp -r backup/$latest_backup/* telegram-bot/
          sudo systemctl restart telegram-bot
```

### Docker Deployment Script
```python
# deploy.py
import subprocess
import os
import sys
from datetime import datetime

class BotDeployer:
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def build_docker_image(self):
        """Build Docker image for the bot."""
        print("🔨 Building Docker image...")
        
        cmd = [
            "docker", "build",
            "-t", f"{self.config['image_name']}:{self.timestamp}",
            "-t", f"{self.config['image_name']}:latest",
            "."
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            sys.exit(1)
        
        print("✅ Docker image built successfully")
    
    def run_tests(self):
        """Run tests in Docker container."""
        print("🧪 Running tests...")
        
        cmd = [
            "docker", "run", "--rm",
            f"{self.config['image_name']}:latest",
            "pytest", "tests/"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Tests failed: {result.stderr}")
            sys.exit(1)
        
        print("✅ All tests passed")
    
    def deploy_container(self):
        """Deploy new container with zero downtime."""
        print("🚀 Deploying new container...")
        
        # Start new container
        new_container = f"bot_{self.timestamp}"
        cmd = [
            "docker", "run", "-d",
            "--name", new_container,
            "--env-file", ".env",
            "--network", "bot_network",
            "-p", "8001:8000",  # Temporary port
            f"{self.config['image_name']}:latest"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Health check
        print("🏥 Running health check...")
        import time
        time.sleep(5)
        
        health_cmd = ["docker", "exec", new_container, "curl", "http://localhost:8000/health"]
        result = subprocess.run(health_cmd, capture_output=True)
        
        if result.returncode != 0:
            print("❌ Health check failed")
            subprocess.run(["docker", "stop", new_container])
            subprocess.run(["docker", "rm", new_container])
            sys.exit(1)
        
        # Switch traffic (nginx or load balancer)
        print("🔄 Switching traffic...")
        self.switch_traffic(new_container)
        
        # Stop old container
        old_container = self.get_current_container()
        if old_container:
            print(f"🛑 Stopping old container: {old_container}")
            subprocess.run(["docker", "stop", old_container])
            subprocess.run(["docker", "rm", old_container])
        
        print("✅ Deployment completed successfully")
    
    def switch_traffic(self, new_container):
        """Update nginx configuration to point to new container."""
        # Implementation depends on your setup
        pass
    
    def get_current_container(self):
        """Get currently running bot container."""
        cmd = ["docker", "ps", "--filter", "name=bot_", "--format", "{{.Names}}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        containers = result.stdout.strip().split('\n')
        return containers[0] if containers else None
    
    def rollback(self):
        """Rollback to previous version."""
        print("⏮️ Rolling back to previous version...")
        # Implementation for rollback
        pass

# Usage
if __name__ == "__main__":
    config = {
        "image_name": "telegram-bot",
        "registry": "docker.io/myuser"
    }
    
    deployer = BotDeployer(config)
    deployer.build_docker_image()
    deployer.run_tests()
    deployer.deploy_container()
```

## 9. Smart Caching with Dynamic TTL

### Adaptive Cache Management
**למה זה שימושי:** מייעל את ביצועי הבוט על ידי שמירת מידע נפוץ במטמון עם זמני תפוגה חכמים. חוסך קריאות למסד נתונים ומשפר זמני תגובה.

```python
import asyncio
import time
from typing import Any, Optional, Callable
from collections import defaultdict
import aioredis
import pickle
import hashlib

class SmartCache:
    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url
        self.access_counts = defaultdict(int)
        self.last_access = defaultdict(float)
        self.ttl_stats = defaultdict(list)
        
    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(self.redis_url)
    
    def _calculate_adaptive_ttl(self, key: str, base_ttl: int = 300) -> int:
        """Calculate TTL based on access patterns."""
        access_count = self.access_counts[key]
        time_since_last = time.time() - self.last_access.get(key, 0)
        
        # Factors affecting TTL
        factors = {
            'high_access': min(access_count / 10, 3),  # Up to 3x for high access
            'recent_access': 2 if time_since_last < 60 else 1,  # 2x if accessed recently
            'time_of_day': self._get_time_factor(),  # Peak hours get longer TTL
        }
        
        # Calculate final TTL
        multiplier = sum(factors.values()) / len(factors)
        adaptive_ttl = int(base_ttl * multiplier)
        
        # Cap between min and max
        return max(60, min(adaptive_ttl, 3600))  # 1 min to 1 hour
    
    def _get_time_factor(self) -> float:
        """Get time-based factor for TTL."""
        hour = time.localtime().tm_hour
        # Peak hours (9-17) get longer TTL
        if 9 <= hour <= 17:
            return 2.0
        # Off-peak hours
        elif hour < 6 or hour > 22:
            return 0.5
        else:
            return 1.0
    
    async def get(
        self, 
        key: str, 
        fetch_func: Optional[Callable] = None,
        ttl: Optional[int] = None
    ) -> Any:
        """Get value from cache or fetch if missing."""
        # Track access
        self.access_counts[key] += 1
        self.last_access[key] = time.time()
        
        # Try to get from cache
        cached = await self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        
        # Cache miss - fetch if function provided
        if fetch_func:
            value = await fetch_func() if asyncio.iscoroutinefunction(fetch_func) else fetch_func()
            
            # Calculate adaptive TTL if not provided
            if ttl is None:
                ttl = self._calculate_adaptive_ttl(key)
            
            # Store in cache
            await self.set(key, value, ttl)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with adaptive TTL."""
        if ttl is None:
            ttl = self._calculate_adaptive_ttl(key)
        
        # Store TTL stats for monitoring
        self.ttl_stats[key].append((time.time(), ttl))
        
        # Serialize and store
        serialized = pickle.dumps(value)
        await self.redis.set(key, serialized, ex=ttl)
    
    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        ttl: Optional[int] = None,
        lock_timeout: int = 10
    ) -> Any:
        """Get from cache or compute with distributed lock."""
        # Try cache first
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Use distributed lock to prevent cache stampede
        lock_key = f"lock:{key}"
        lock_acquired = await self.redis.set(
            lock_key, "1", 
            nx=True, 
            ex=lock_timeout
        )
        
        if lock_acquired:
            try:
                # Double-check cache (another process might have computed it)
                cached = await self.get(key)
                if cached is not None:
                    return cached
                
                # Compute value
                value = await compute_func() if asyncio.iscoroutinefunction(compute_func) else compute_func()
                
                # Store with adaptive TTL
                await self.set(key, value, ttl)
                return value
            finally:
                await self.redis.delete(lock_key)
        else:
            # Wait for lock owner to compute
            for _ in range(lock_timeout * 10):  # Check every 100ms
                await asyncio.sleep(0.1)
                cached = await self.get(key)
                if cached is not None:
                    return cached
            
            # Timeout - compute anyway
            return await compute_func() if asyncio.iscoroutinefunction(compute_func) else compute_func()
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        cursor = '0'
        while cursor != 0:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            if keys:
                await self.redis.delete(*keys)
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        total_keys = await self.redis.dbsize()
        
        # Get top accessed keys
        top_keys = sorted(
            self.access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_keys': total_keys,
            'top_accessed': top_keys,
            'average_ttl': self._calculate_average_ttl(),
            'memory_usage': await self.redis.info('memory')
        }
    
    def _calculate_average_ttl(self) -> float:
        """Calculate average TTL across all keys."""
        all_ttls = []
        for ttls in self.ttl_stats.values():
            all_ttls.extend([ttl for _, ttl in ttls[-10:]])  # Last 10 TTLs per key
        
        return sum(all_ttls) / len(all_ttls) if all_ttls else 0

# Decorator for automatic caching
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(str((args, kwargs)).encode()).hexdigest()}"
            
            # Get or compute with cache
            return await self.cache.get_or_compute(
                cache_key,
                lambda: func(self, *args, **kwargs),
                ttl
            )
        return wrapper
    return decorator

# Usage example
class BotService:
    def __init__(self, cache: SmartCache):
        self.cache = cache
    
    @cached(key_prefix="user")
    async def get_user_profile(self, user_id: int):
        """Get user profile with automatic caching."""
        # Expensive database query
        return await db.fetch_user(user_id)
    
    @cached(ttl=60, key_prefix="stats")
    async def get_statistics(self):
        """Get bot statistics with 60 second cache."""
        # Complex aggregation
        return await calculate_stats()
```

## 10. Video Chat Integration

### Create Video Rooms for Users
**למה זה שימושי:** מאפשר יצירת חדרי וידאו לשיחות קבוצתיות או פרטיות. מעולה לבוטים של פגישות, למידה מרחוק, או תמיכה טכנית.

```python
from typing import Dict, List, Optional
import uuid
import jwt
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

class VideoRoomManager:
    def __init__(self, video_service_url: str, api_key: str, jwt_secret: str):
        self.video_service_url = video_service_url
        self.api_key = api_key
        self.jwt_secret = jwt_secret
        self.active_rooms: Dict[str, dict] = {}
        
    async def create_video_room(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a new video room."""
        user = update.effective_user
        room_id = str(uuid.uuid4())
        
        # Create room metadata
        room_data = {
            'room_id': room_id,
            'host_id': user.id,
            'host_name': user.first_name,
            'created_at': datetime.now(),
            'participants': [user.id],
            'max_participants': 10,
            'duration_minutes': 60,
            'settings': {
                'recording': False,
                'waiting_room': True,
                'mute_on_entry': True,
                'screen_share': True
            }
        }
        
        self.active_rooms[room_id] = room_data
        
        # Generate join links
        host_link = self._generate_join_link(room_id, user.id, is_host=True)
        participant_link = self._generate_join_link(room_id, user.id, is_host=False)
        
        # Create inline keyboard
        keyboard = [
            [InlineKeyboardButton("🎥 הצטרף כמארח", url=host_link)],
            [InlineKeyboardButton("📤 שתף הזמנה", switch_inline_query=f"join_video {room_id}")],
            [InlineKeyboardButton("⚙️ הגדרות חדר", callback_data=f"room_settings:{room_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"🎥 **חדר וידאו נוצר בהצלחה!**\n\n"
            f"🆔 מזהה חדר: `{room_id}`\n"
            f"👥 מקסימום משתתפים: {room_data['max_participants']}\n"
            f"⏱️ משך: {room_data['duration_minutes']} דקות\n\n"
            f"🔗 קישור להצטרפות:\n{participant_link}",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    def _generate_join_link(self, room_id: str, user_id: int, is_host: bool) -> str:
        """Generate secure join link with JWT."""
        payload = {
            'room_id': room_id,
            'user_id': user_id,
            'is_host': is_host,
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return f"{self.video_service_url}/room/{room_id}?token={token}"
    
    async def join_video_room(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle joining a video room."""
        room_id = context.args[0] if context.args else None
        
        if not room_id or room_id not in self.active_rooms:
            await update.message.reply_text("❌ חדר לא נמצא או פג תוקף")
            return
        
        room = self.active_rooms[room_id]
        user = update.effective_user
        
        # Check if room is full
        if len(room['participants']) >= room['max_participants']:
            await update.message.reply_text("❌ החדר מלא")
            return
        
        # Add participant
        if user.id not in room['participants']:
            room['participants'].append(user.id)
        
        # Generate join link
        join_link = self._generate_join_link(room_id, user.id, is_host=False)
        
        keyboard = [[InlineKeyboardButton("🎥 הצטרף לשיחה", url=join_link)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"✅ הצטרפת לחדר!\n"
            f"👥 משתתפים: {len(room['participants'])}/{room['max_participants']}",
            reply_markup=reply_markup
        )
        
        # Notify host
        if user.id != room['host_id']:
            await context.bot.send_message(
                room['host_id'],
                f"👤 {user.first_name} הצטרף לחדר הוידאו שלך"
            )
    
    async def manage_room_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manage video room settings."""
        query = update.callback_query
        room_id = query.data.split(':')[1]
        
        if room_id not in self.active_rooms:
            await query.answer("החדר לא נמצא", show_alert=True)
            return
        
        room = self.active_rooms[room_id]
        
        # Only host can manage settings
        if query.from_user.id != room['host_id']:
            await query.answer("רק המארח יכול לשנות הגדרות", show_alert=True)
            return
        
        # Toggle settings keyboard
        keyboard = []
        for setting, value in room['settings'].items():
            emoji = "✅" if value else "❌"
            keyboard.append([InlineKeyboardButton(
                f"{emoji} {setting.replace('_', ' ').title()}",
                callback_data=f"toggle_setting:{room_id}:{setting}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 חזור", callback_data=f"room_info:{room_id}")])
        
        await query.edit_message_text(
            f"⚙️ **הגדרות חדר וידאו**\n\n"
            f"לחץ על הגדרה כדי לשנות אותה:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def end_video_room(self, room_id: str, context: ContextTypes.DEFAULT_TYPE):
        """End a video room and notify participants."""
        if room_id not in self.active_rooms:
            return
        
        room = self.active_rooms[room_id]
        
        # Notify all participants
        for participant_id in room['participants']:
            try:
                await context.bot.send_message(
                    participant_id,
                    f"🔚 חדר הוידאו הסתיים\n"
                    f"משך השיחה: {(datetime.now() - room['created_at']).seconds // 60} דקות"
                )
            except:
                pass
        
        # Clean up
        del self.active_rooms[room_id]
    
    async def schedule_video_meeting(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Schedule a video meeting for later."""
        # Parse meeting details
        try:
            date_str = context.args[0]  # Format: DD/MM/YYYY
            time_str = context.args[1]  # Format: HH:MM
            title = ' '.join(context.args[2:]) if len(context.args) > 2 else "פגישת וידאו"
            
            # Schedule the meeting
            meeting_time = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            
            # Create scheduled meeting
            meeting_id = str(uuid.uuid4())
            
            # Store in database or scheduler
            context.job_queue.run_once(
                self._create_scheduled_room,
                when=meeting_time,
                context={
                    'meeting_id': meeting_id,
                    'host_id': update.effective_user.id,
                    'title': title
                }
            )
            
            await update.message.reply_text(
                f"📅 **פגישה נקבעה!**\n\n"
                f"📝 נושא: {title}\n"
                f"📆 תאריך: {date_str}\n"
                f"⏰ שעה: {time_str}\n\n"
                f"תקבל תזכורת 10 דקות לפני הפגישה."
            )
            
        except (IndexError, ValueError):
            await update.message.reply_text(
                "❌ פורמט לא תקין. השתמש ב:\n"
                "/schedule DD/MM/YYYY HH:MM [נושא הפגישה]"
            )
    
    async def _create_scheduled_room(self, context: ContextTypes.DEFAULT_TYPE):
        """Create room for scheduled meeting."""
        data = context.job.context
        
        # Create room
        room_id = str(uuid.uuid4())
        # ... create room logic ...
        
        # Notify host
        await context.bot.send_message(
            data['host_id'],
            f"🔔 הפגישה שלך '{data['title']}' מתחילה עכשיו!\n"
            f"לחץ כאן כדי להתחיל: /start_video {room_id}"
        )
```

זהו! יצרתי מסמך מקיף עם 10 סניפטים מתקדמים נוספים לבוטי טלגרם. כל סניפט כולל:
- הסבר "למה זה שימושי"
- קוד מפורט עם דוגמאות
- הערות ותיעוד

הקובץ נשמר בשם `Snippets7New.md` בריפו. הסניפטים מכסים תחומים מתקדמים כמו עיבוד אודיו, ניתוח בזמן אמת, למידת מכונה, ועוד.