# 📚 ספריית Code Snippets – Telegram Bot Development

ספריית תבניות קוד שימושיות למפתחי בוטים בטלגרם, מופקות מהפרויקט Markdown Trainer Bot.

---

## 21.x – Telegram Bot Patterns

### 21.1 – Maintenance Mode Middleware

**למה זה שימושי:** מאפשר להכניס את הבוט למצב תחזוקה תוך שמירה על גישה לאדמינים. תבנית middleware פשוטה וחזקה.

```javascript
const isMaintenanceMode = () => {
  return process.env.MAINTENANCE_MODE === 'true';
};

const isAdmin = (userId) => {
  const admins = (process.env.ADMIN_USER_IDS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  return admins.includes(String(userId));
};

const maintenanceMiddleware = async (msg, next) => {
  if (!isMaintenanceMode() || isAdmin(msg.from.id)) {
    return next();
  }

  await bot.sendMessage(msg.chat.id,
    '🔧 *הבוט במצב תחזוקה*\n\n' +
    'אנחנו עושים שדרוג מהיר לבוט 🚀\n\n' +
    'הכל יחזור לעבוד תוך כמה דקות.\n\n' +
    'תודה על הסבלנות! 😊',
    { parse_mode: 'Markdown' }
  );
  return false; // Block further processing
};

// שימוש:
bot.onText(/\/start/, async (msg) => {
  if (await maintenanceMiddleware(msg, () => true)) {
    commandHandler.handleStart(msg);
  }
});
```

---

### 21.2 – Callback Query Router

**למה זה שימושי:** ניתוב אוטומטי של callback queries לפי prefix, חוסך if-else ארוכים ומארגן את הקוד.

```javascript
async handleCallbackQuery(query) {
  const chatId = query.message.chat.id;
  const userId = query.from.id;
  const data = query.data;
  const messageId = query.message.message_id;

  // Answer callback to remove loading state
  await this.bot.answerCallbackQuery(query.id);

  // Route based on callback data prefix
  if (data.startsWith('pace_')) {
    await this.handlePaceSelection(chatId, userId, data, messageId);
  } else if (data.startsWith('answer_')) {
    await this.handleQuizAnswer(chatId, userId, data, messageId);
  } else if (data.startsWith('theme_')) {
    await this.handleThemeSelection(chatId, userId, data, messageId);
  } else if (data.startsWith('cheat_')) {
    await this.handleCheatsheetTopic(chatId, userId, data, messageId);
  } else if (data.startsWith('template_')) {
    await this.handleTemplateSelection(chatId, userId, data);
  }
  // ... more routes
}
```

---

### 21.3 – User Mode Management (State Machine)

**למה זה שימושי:** ניהול מצבי משתמש (sandbox, training, submitting_template) עם נתונים JSON. מאפשר זרימות מורכבות.

```javascript
// Database schema:
// CREATE TABLE user_modes (
//   user_id INTEGER PRIMARY KEY,
//   current_mode TEXT DEFAULT 'normal',
//   mode_data TEXT,  -- JSON string
//   updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
// )

getUserMode(userId) {
  const stmt = this.db.prepare('SELECT * FROM user_modes WHERE user_id = ?');
  let mode = stmt.get(userId);
  
  if (!mode) {
    const insertStmt = this.db.prepare(`
      INSERT INTO user_modes (user_id, current_mode)
      VALUES (?, 'normal')
    `);
    insertStmt.run(userId);
    mode = { user_id: userId, current_mode: 'normal', mode_data: null };
  }
  
  return mode;
}

setUserMode(userId, mode, modeData = null) {
  const stmt = this.db.prepare(`
    INSERT OR REPLACE INTO user_modes (user_id, current_mode, mode_data, updated_at)
    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
  `);
  stmt.run(userId, mode, modeData ? JSON.stringify(modeData) : null);
}

clearUserMode(userId) {
  this.setUserMode(userId, 'normal', null);
}

// שימוש:
const mode = this.db.getUserMode(userId);
if (mode.current_mode === 'sandbox') {
  await this.handleSandboxInput(chatId, userId, text);
} else if (mode.current_mode === 'training') {
  const modeData = JSON.parse(mode.mode_data);
  await this.handleTrainingAnswer(chatId, userId, text, modeData);
}
```

---

### 21.4 – Message Chunking (Split Long Messages)

**למה זה שימושי:** טלגרם מגביל הודעות ל-4096 תווים. תבנית זו מחלקת הודעות ארוכות לחתיכות תוך שמירה על קריאות.

```javascript
async sendLongMessage(chatId, text, options = {}) {
  const maxLength = 4000; // Leave room for formatting
  const chunks = [];
  let currentChunk = '';
  const lines = text.split('\n');

  for (const line of lines) {
    if ((currentChunk + line + '\n').length > maxLength) {
      chunks.push(currentChunk);
      currentChunk = line + '\n';
    } else {
      currentChunk += line + '\n';
    }
  }
  if (currentChunk) chunks.push(currentChunk);

  // Send each chunk
  for (let i = 0; i < chunks.length; i++) {
    await this.bot.sendMessage(chatId,
      chunks.length > 1 ? `*חלק ${i + 1}/${chunks.length}*\n\n${chunks[i]}` : chunks[i],
      { parse_mode: 'Markdown', ...options }
    );
    await this.sleep(500); // Small delay between messages
  }
}
```

---

### 21.5 – Progress Bar Generation

**למה זה שימושי:** יצירת progress bar טקסטואלי יפה להודעות טלגרם, שימושי להצגת התקדמות.

```javascript
createProgressBar(current, total, barLength = 10) {
  const filledBars = Math.floor((current / total) * barLength);
  const emptyBars = barLength - filledBars;
  return '█'.repeat(filledBars) + '░'.repeat(emptyBars);
}

// שימוש:
const progress = this.db.getUserProgress(userId);
const totalLessons = 14;
const progressBar = this.createProgressBar(
  progress.lessons_completed,
  totalLessons,
  10
);
const message = `📈 *התקדמות בשיעורים:*\n${progressBar} ${progressPercentage}%\n${progress.lessons_completed}/${totalLessons} שיעורים הושלמו`;
```

---

### 21.6 – Safe Send with Fallback

**למה זה שימושי:** שליחת הודעות Markdown עם fallback אוטומטי לטקסט רגיל אם יש שגיאת parsing. חוסך קריסות.

```javascript
async safeSendMarkdown(chatId, text, options = {}) {
  try {
    return await this.bot.sendMessage(chatId, text, { 
      parse_mode: 'Markdown', 
      ...options 
    });
  } catch (err) {
    const desc = String(err?.response?.body?.description || err?.message || '').toLowerCase();
    if (desc.includes("can't parse entities") || desc.includes('parse') || desc.includes('entity')) {
      // Retry without parse mode
      return await this.bot.sendMessage(chatId, text, { ...options });
    }
    throw err;
  }
}

// גרסה ל-MarkdownV2:
async safeSendMarkdownV2(chatId, text, options = {}) {
  try {
    return await this.bot.sendMessage(chatId, text, {
      parse_mode: 'MarkdownV2',
      disable_web_page_preview: true,
      ...options,
    });
  } catch (err) {
    const desc = String(err?.response?.body?.description || err?.message || '').toLowerCase();
    if (desc.includes("can't parse entities") || desc.includes('parse') || desc.includes('entity')) {
      // Remove language hints from code fences
      const withoutLangFences = text.replace(/```[a-zA-Z0-9_+#-]+/g, '```');
      if (withoutLangFences !== text) {
        try {
          return await this.bot.sendMessage(chatId, withoutLangFences, {
            parse_mode: 'MarkdownV2',
            disable_web_page_preview: true,
            ...options,
          });
        } catch (_) {
          // fall through to plain text fallback
        }
      }
      const plain = this.unescapeMarkdownV2(text);
      return await this.bot.sendMessage(chatId, plain, { 
        disable_web_page_preview: true, 
        ...options 
      });
    }
    throw err;
  }
}
```

---

### 21.7 – Markdown Reconstruction from Entities

**למה זה שימושי:** שחזור Markdown מה-entities של טלגרם. מאפשר למשתמשים לשלוח טקסט מעוצב (bold, italic) והבוט יזהה את הסימונים.

```javascript
reconstructMarkdownFromEntities(text, entities = []) {
  if (!text || !Array.isArray(entities) || entities.length === 0) {
    return text;
  }

  const openMap = Object.create(null);
  const closeMap = Object.create(null);

  const getPriority = (type) => {
    switch (type) {
      case 'pre': return 0;
      case 'code': return 10;
      case 'bold': return 20;
      case 'italic': return 30;
      case 'underline': return 40;
      case 'strikethrough': return 50;
      case 'text_link': return 60;
      default: return 100;
    }
  };

  for (const entity of entities) {
    if (!entity || typeof entity.offset !== 'number' || typeof entity.length !== 'number') {
      continue;
    }

    const start = entity.offset;
    const end = entity.offset + entity.length;

    if (start < 0 || end > text.length || start >= end) {
      continue;
    }

    let openWrapper = '';
    let closeWrapper = '';

    switch (entity.type) {
      case 'bold':
        openWrapper = '**';
        closeWrapper = '**';
        break;
      case 'italic':
        openWrapper = '*';
        closeWrapper = '*';
        break;
      case 'code':
        openWrapper = '`';
        closeWrapper = '`';
        break;
      case 'pre': {
        const language = entity.language ? String(entity.language).trim() : '';
        openWrapper = '```' + (language || '') + '\n';
        closeWrapper = '\n```';
        break;
      }
      case 'text_link':
        if (entity.url) {
          openWrapper = '[';
          closeWrapper = `](${entity.url})`;
        }
        break;
      default:
        break;
    }

    if (!openWrapper && !closeWrapper) {
      continue;
    }

    const priority = getPriority(entity.type);

    if (!openMap[start]) openMap[start] = [];
    if (!closeMap[end]) closeMap[end] = [];

    openMap[start].push({ text: openWrapper, priority });
    closeMap[end].push({ text: closeWrapper, priority });
  }

  let result = '';

  for (let i = 0; i < text.length; i++) {
    if (closeMap[i]) {
      closeMap[i].sort((a, b) => b.priority - a.priority);
      for (const closer of closeMap[i]) {
        result += closer.text;
      }
    }

    if (openMap[i]) {
      openMap[i].sort((a, b) => a.priority - b.priority);
      for (const opener of openMap[i]) {
        result += opener.text;
      }
    }

    result += text[i];
  }

  if (closeMap[text.length]) {
    closeMap[text.length].sort((a, b) => b.priority - a.priority);
    for (const closer of closeMap[text.length]) {
      result += closer.text;
    }
  }

  return result;
}

// שימוש:
const normalizedText = this.reconstructMarkdownFromEntities(msg.text, msg.entities);
```

---

### 21.8 – Inline Keyboard Builder

**למה זה שימושי:** בניית inline keyboards דינמיים בקלות, שימושי לתפריטים, בחירות, וניווט.

```javascript
buildInlineKeyboard(buttons) {
  // buttons: [[{text, callback_data}], [{text, callback_data}]]
  return {
    inline_keyboard: buttons
  };
}

// דוגמה: תפריט תבניות
const keyboard = [
  [{ text: '📋 PRD - מסמך דרישות', callback_data: 'template_prd' }],
  [{ text: '📖 README - תיעוד פרויקט', callback_data: 'template_readme' }],
  [{ text: '🔍 Post Mortem - ניתוח תקלה', callback_data: 'template_postmortem' }]
];

// הוספת תבניות קהילתיות דינמית
const communityTemplates = this.db.getCommunityTemplates();
if (communityTemplates.length > 0) {
  keyboard.push([{ text: '━━━━ תבניות קהילתיות ━━━━', callback_data: 'noop' }]);
  communityTemplates.forEach(template => {
    keyboard.push([{
      text: `👥 ${template.title} (${template.first_name || 'קהילה'})`,
      callback_data: `community_template_${template.template_id}`
    }]);
  });
}

await this.bot.sendMessage(chatId, 'בחר תבנית:', {
  reply_markup: { inline_keyboard: keyboard }
});
```

---

### 21.9 – Multi-Step Form Flow

**למה זה שימושי:** זרימת הגשת תבניות/טפסים רב-שלבית. שימושי לשאלונים, הרשמות, והגשות.

```javascript
// שלב 1: התחלה
async handleSubmitTemplate(msg) {
  const userId = msg.from.id;
  this.db.setUserMode(userId, 'submitting_template', JSON.stringify({ step: 'title' }));
  
  await this.bot.sendMessage(chatId,
    '📝 *שלב 1 מתוך 4: כותרת*\n' +
    'מה שם התבנית?',
    { parse_mode: 'Markdown' }
  );
}

// שלב 2: טיפול בקלט
async handleTemplateSubmissionInput(chatId, userId, text, mode) {
  const modeData = JSON.parse(mode.mode_data || '{}');
  const step = modeData.step;

  if (step === 'title') {
    // Validate
    if (text.length < 3 || text.length > 100) {
      await this.bot.sendMessage(chatId, '❌ הכותרת צריכה להיות בין 3 ל-100 תווים.');
      return;
    }

    // Save and advance
    modeData.title = text;
    modeData.step = 'category';
    this.db.setUserMode(userId, 'submitting_template', JSON.stringify(modeData));

    await this.bot.sendMessage(chatId,
      `✅ כותרת נשמרה: "${text}"\n\n` +
      `📝 *שלב 2 מתוך 4: קטגוריה*`,
      { parse_mode: 'Markdown' }
    );
  } else if (step === 'category') {
    // ... continue flow
  } else if (step === 'content') {
    // Final step - save and complete
    this.db.createTemplateSubmission(userId, templateId, modeData.title, ...);
    this.db.clearUserMode(userId);
    await this.bot.sendMessage(chatId, '🎉 תבנית נשלחה בהצלחה!');
  }
}
```

---

### 21.10 – Statistics Aggregation

**למה זה שימושי:** איגום סטטיסטיקות מורכבות עם JOINs ו-subqueries. שימושי לדשבורדים ומסכי ניהול.

```javascript
getUserActivityStats(days = 7) {
  const stmt = this.db.prepare(`
    SELECT
      u.user_id,
      u.username,
      u.first_name,
      u.last_name,
      u.last_active,
      up.lessons_completed,
      up.correct_answers,
      up.wrong_answers,
      up.total_score,
      up.level,
      (SELECT COUNT(*) FROM lesson_history lh
       WHERE lh.user_id = u.user_id
       AND lh.completed_at >= datetime('now', '-' || ? || ' days')) as recent_lessons,
      (SELECT COUNT(*) FROM training_sessions ts
       WHERE ts.user_id = u.user_id
       AND ts.started_at >= datetime('now', '-' || ? || ' days')) as recent_training_sessions
    FROM users u
    LEFT JOIN user_progress up ON u.user_id = up.user_id
    WHERE u.last_active >= datetime('now', '-' || ? || ' days')
    ORDER BY recent_lessons DESC, u.last_active DESC
  `);
  return stmt.all(days, days, days);
}

// שימוש:
const userStats = this.db.getUserActivityStats(7);
for (const user of userStats) {
  const totalActions = (user.recent_lessons || 0) + (user.recent_training_sessions || 0);
  const accuracy = totalAnswers > 0 ? 
    ((user.correct_answers / totalAnswers) * 100).toFixed(1) : 0;
  // ... format and send
}
```

---

## 22.x – Database Patterns

### 22.1 – Database Transaction Pattern

**למה זה שימושי:** ביצוע מספר פעולות DB כטרנזקציה אטומית. מבטיח עקביות נתונים.

```javascript
resetUserProgress(userId) {
  const trx = this.db.transaction((uid) => {
    // Reset core progress fields
    const resetStmt = this.db.prepare(`
      UPDATE user_progress
      SET current_lesson = 0,
          total_score = 0,
          level = 'Beginner',
          lessons_completed = 0,
          correct_answers = 0,
          wrong_answers = 0,
          last_lesson_date = NULL
      WHERE user_id = ?
    `);
    resetStmt.run(uid);

    // Clear lesson history and topic performance
    const delHistory = this.db.prepare('DELETE FROM lesson_history WHERE user_id = ?');
    delHistory.run(uid);
    const delTopics = this.db.prepare('DELETE FROM topic_performance WHERE user_id = ?');
    delTopics.run(uid);

    // Reset user mode to normal
    const resetMode = this.db.prepare(`
      INSERT OR REPLACE INTO user_modes (user_id, current_mode, mode_data, updated_at)
      VALUES (?, 'normal', NULL, CURRENT_TIMESTAMP)
    `);
    resetMode.run(uid);
  });

  trx(userId);
}
```

---

### 22.2 – Upsert with ON CONFLICT

**למה זה שימושי:** עדכון או הוספת רשומה בקריאה אחת. שימושי למעקב ביצועים, העדפות משתמש.

```javascript
updateTopicPerformance(userId, topic, isCorrect) {
  const field = isCorrect ? 'correct_count' : 'wrong_count';
  
  const stmt = this.db.prepare(`
    INSERT INTO topic_performance (user_id, topic, ${field}, last_practiced)
    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
    ON CONFLICT(user_id, topic) DO UPDATE SET
      ${field} = ${field} + 1,
      last_practiced = CURRENT_TIMESTAMP
  `);
  
  stmt.run(userId, topic);
}
```

---

### 22.3 – Database Initialization with Migrations

**למה זה שימושי:** יצירת טבלאות עם תמיכה ב-migrations. מאפשר הוספת עמודות חדשות בבטחה.

```javascript
initializeTables() {
  // Create main table
  this.db.exec(`
    CREATE TABLE IF NOT EXISTS user_progress (
      user_id INTEGER PRIMARY KEY,
      current_lesson INTEGER DEFAULT 0,
      total_score INTEGER DEFAULT 0,
      level TEXT DEFAULT 'Beginner'
    )
  `);

  // Migration: Add new column if it doesn't exist
  try {
    this.db.exec(`
      ALTER TABLE user_progress ADD COLUMN sandbox_theme TEXT DEFAULT 'github-light'
    `);
    console.log('✅ Added sandbox_theme column');
  } catch (error) {
    // Column already exists, ignore the error
    if (!error.message.includes('duplicate column')) {
      console.warn('⚠️ Migration warning:', error.message);
    }
  }
}
```

---

## 23.x – Puppeteer & Browser Management

### 23.1 – Puppeteer Browser Singleton

**למה זה שימושי:** ניהול instance יחיד של browser, חוסך זמן פתיחה/סגירה. שימושי לרינדור תמונות/PDFs.

```javascript
class MarkdownRenderer {
  constructor() {
    this.browser = null;
  }

  async initBrowser() {
    if (!this.browser) {
      this.browser = await puppeteer.launch({
        headless: 'new',
        executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || puppeteer.executablePath(),
        args: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
          '--disable-accelerated-2d-canvas',
          '--disable-gpu'
        ]
      });
      console.log('✅ Puppeteer browser initialized');
    }
    return this.browser;
  }

  async closeBrowser() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      console.log('✅ Puppeteer browser closed');
    }
  }

  async renderMarkdown(markdownText, userId, theme) {
    await this.initBrowser();
    const page = await this.browser.newPage();
    // ... render logic
    await page.close();
    // Don't close browser - reuse for next request
  }
}
```

---

### 23.2 – Dynamic Viewport Adjustment

**למה זה שימושי:** התאמת גובה viewport לתוכן. חוסך מקום ונותן screenshots מדויקים.

```javascript
async renderWithDynamicHeight(page, html) {
  // Set initial viewport
  await page.setViewport({
    width: 800,
    height: 600,
    deviceScaleFactor: 2
  });

  await page.setContent(html, { waitUntil: 'networkidle0' });

  // Get content height
  const contentHeight = await page.evaluate(() => {
    return document.querySelector('.markdown-body').scrollHeight;
  });

  // Adjust viewport to content
  await page.setViewport({
    width: 800,
    height: Math.min(contentHeight + 40, 4000), // Max 4000px
    deviceScaleFactor: 2
  });

  // Take screenshot
  await page.screenshot({
    path: outputPath,
    fullPage: true
  });
}
```

---

## 24.x – File Management

### 24.1 – File Cleanup Pattern

**למה זה שימושי:** ניקוי אוטומטי של קבצים ישנים. מונע התמלאות דיסק.

```javascript
cleanupOldFiles(userId, maxAge = 24 * 60 * 60 * 1000) {
  try {
    const files = fs.readdirSync(this.outputDir);
    const now = Date.now();

    files.forEach(file => {
      const filePath = path.join(this.outputDir, file);
      const stats = fs.statSync(filePath);
      const age = now - stats.mtime.getTime();

      if (age > maxAge) {
        fs.unlinkSync(filePath);
        console.log(`🗑️ Deleted old file: ${file}`);
      }
    });
  } catch (error) {
    console.error('Error cleaning up old files:', error);
  }
}

// Keep only last N files per user
cleanupUserFiles(userId, keepCount = 5) {
  try {
    const files = fs.readdirSync(this.outputDir);
    const userFiles = files.filter(f => f.includes(`markdown_${userId}_`));
    
    if (userFiles.length > keepCount) {
      const sortedFiles = userFiles
        .map(f => ({
          name: f,
          time: fs.statSync(path.join(this.outputDir, f)).mtime.getTime()
        }))
        .sort((a, b) => a.time - b.time);

      const filesToDelete = sortedFiles.slice(0, sortedFiles.length - keepCount);
      filesToDelete.forEach(file => {
        fs.unlinkSync(path.join(this.outputDir, file.name));
      });
    }
  } catch (error) {
    console.error('Error cleaning up user files:', error);
  }
}
```

---

### 24.2 – Safe File Operations

**למה זה שימושי:** יצירת תיקיות וקבצים עם טיפול בשגיאות. מונע קריסות.

```javascript
ensureDirectoryExists(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
      console.log(`📁 Created directory: ${dirPath}`);
    }
  } catch (error) {
    console.error(`❌ Failed to create directory ${dirPath}:`, error);
    throw error;
  }
}

// שימוש:
this.outputDir = path.join(__dirname, '../output');
this.ensureDirectoryExists(this.outputDir);
```

---

## 25.x – Utility Functions

### 25.1 – Relative Time Formatting

**למה זה שימושי:** הצגת זמן יחסי בעברית (לפני X דקות/שעות/ימים). משפר UX.

```javascript
formatRelativeTime(dateString) {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'כרגע';
  if (diffMins < 60) return `לפני ${diffMins} דקות`;
  if (diffHours < 24) return `לפני ${diffHours} שעות`;
  if (diffDays < 7) return `לפני ${diffDays} ימים`;
  return date.toLocaleDateString('he-IL');
}

// שימוש:
const lastActive = this.formatRelativeTime(user.last_active);
// "לפני 5 דקות" / "לפני 2 שעות" / "כרגע"
```

---

### 25.2 – HTML Entity Escaping

**למה זה שימושי:** Escape של HTML entities לפני שליחה ב-parse_mode=HTML. מונע XSS ובעיות parsing.

```javascript
escapeHtmlEntities(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

stripHtmlTags(html) {
  if (!html) return '';
  return String(html).replace(/<[^>]*>/g, '');
}

// שימוש:
const safeHtml = this.escapeHtmlEntities(userInput);
await this.bot.sendMessage(chatId, safeHtml, { parse_mode: 'HTML' });
```

---

### 25.3 – Sleep/Delay Helper

**למה זה שימושי:** השהיה בין הודעות. משפר UX ומונע rate limiting.

```javascript
sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// שימוש:
await this.bot.sendMessage(chatId, 'הודעה ראשונה');
await this.sleep(1500); // 1.5 seconds
await this.bot.sendMessage(chatId, 'הודעה שנייה');
```

---

### 25.4 – Admin Check Helper

**למה זה שימושי:** בדיקה מהירה אם משתמש הוא אדמין. שימושי להגנת פקודות.

```javascript
isAdmin(userId) {
  const admins = (process.env.ADMIN_USER_IDS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  return admins.includes(String(userId));
}

// שימוש:
async handleAdminCommand(msg) {
  if (!this.isAdmin(msg.from.id)) {
    await this.bot.sendMessage(msg.chat.id, '⛔ פקודה זו זמינה רק לאדמין.');
    return;
  }
  // ... admin logic
}
```

---

## 26.x – Error Handling & Resilience

### 26.1 – Graceful Shutdown

**למה זה שימושי:** כיבוי מסודר של הבוט. סוגר connections, שומר נתונים, מונע data loss.

```javascript
const gracefulShutdown = () => {
  console.log('\n🛑 Shutting down gracefully...');
  
  if (bot) {
    bot.stopPolling();
  }
  db.close();
  
  process.exit(0);
};

process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);

// Error handlers
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});
```

---

### 26.2 – Error Handling with User Feedback

**למה זה שימושי:** טיפול בשגיאות עם הודעות ידידותיות למשתמש. משפר UX.

```javascript
async handleSandboxInput(chatId, userId, markdownText) {
  try {
    const processingMsg = await this.bot.sendMessage(chatId, '⏳ מעבד את הקוד שלך...');
    const imagePath = await this.renderer.renderMarkdown(markdownText, userId, theme);
    await this.bot.deleteMessage(chatId, processingMsg.message_id);
    await this.bot.sendPhoto(chatId, imagePath, {
      caption: '✅ הנה התוצאה המעוצבת של הקוד שלך!'
    });
  } catch (error) {
    console.error('Error rendering markdown:', error);
    await this.bot.sendMessage(chatId,
      '❌ אופס! משהו השתבש בעיבוד הקוד.\n\n' +
      'ייתכן שהקוד ארוך מדי או מכיל תווים לא נתמכים.\n\n' +
      'נסה שוב או שלח /exit לצאת מהמעבדה.'
    );
  }
}
```

---

## 27.x – Environment & Configuration

### 27.1 – Environment Variable Validation

**למה זה שימושי:** בדיקת משתני סביבה חיוניים בהפעלה. מונע שגיאות runtime.

```javascript
const token = process.env.TELEGRAM_BOT_TOKEN;
let bot = null;

if (!token) {
  console.warn('⚠️ לא הוגדר TELEGRAM_BOT_TOKEN — השירות יעלה במצב בריאות בלבד (ללא בוט)');
} else {
  bot = new TelegramBot(token, { polling: true });
}

// Database path with fallback
let dbPath = process.env.DATABASE_PATH || '/data/users.db';
const dbDir = path.dirname(dbPath);

try {
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }
} catch (error) {
  console.warn(`⚠️ Cannot create directory ${dbDir}:`, error.message);
  dbPath = path.join(__dirname, 'users.db'); // Fallback to local
}
```

---

### 27.2 – Puppeteer Executable Path Discovery

**למה זה שימושי:** גילוי אוטומטי של נתיב Chrome/Puppeteer. שימושי בסביבות deploy שונות.

```javascript
if (!process.env.PUPPETEER_EXECUTABLE_PATH) {
  try {
    const baseCacheDir = process.env.PUPPETEER_CACHE_DIR || '/opt/render/.cache/puppeteer';
    const chromeRoot = path.join(baseCacheDir, 'chrome');
    let discoveredExecutablePath = '';
    
    if (fs.existsSync(chromeRoot)) {
      const platformPrefixes = ['linux-', 'mac-', 'win-'];
      const versions = fs
        .readdirSync(chromeRoot)
        .filter((entry) => platformPrefixes.some((p) => entry.startsWith(p)));
      
      // Newest first by mtime
      const versionEntriesSorted = versions
        .map((entry) => ({
          name: entry,
          time: fs.statSync(path.join(chromeRoot, entry)).mtimeMs,
        }))
        .sort((a, b) => b.time - a.time);
      
      for (const v of versionEntriesSorted) {
        const candidate = path.join(chromeRoot, v.name, 'chrome-linux64', 'chrome');
        if (fs.existsSync(candidate)) {
          discoveredExecutablePath = candidate;
          break;
        }
      }
    }
    
    if (discoveredExecutablePath) {
      process.env.PUPPETEER_EXECUTABLE_PATH = discoveredExecutablePath;
      console.log(`[boot] Using Puppeteer Chrome at: ${discoveredExecutablePath}`);
    }
  } catch (e) {
    console.log('[boot] Failed to resolve Puppeteer executable path:', e && e.message);
  }
}
```

---

## 28.x – Advanced Patterns

### 28.1 – Processing Message with Loading Indicator

**למה זה שימושי:** הצגת הודעת "מעבד..." בזמן עיבוד ארוך. משפר UX.

```javascript
async handleSandboxInput(chatId, userId, markdownText) {
  // Send "processing" message
  const processingMsg = await this.bot.sendMessage(chatId, '⏳ מעבד את הקוד שלך...');

  try {
    // Long-running operation
    const imagePath = await this.renderer.renderMarkdown(markdownText, userId, theme);

    // Delete processing message
    await this.bot.deleteMessage(chatId, processingMsg.message_id);

    // Send result
    await this.bot.sendPhoto(chatId, imagePath, {
      caption: '✅ הנה התוצאה!'
    });
  } catch (error) {
    // Delete processing message even on error
    try {
      await this.bot.deleteMessage(chatId, processingMsg.message_id);
    } catch (_) {}
    
    await this.bot.sendMessage(chatId, '❌ שגיאה בעיבוד.');
  }
}
```

---

### 28.2 – Reply Keyboard (Main Menu)

**למה זה שימושי:** תפריט ראשי קבוע בתחתית המסך. משפר ניווט.

```javascript
getMainKeyboard() {
  return {
    keyboard: [
      [{ text: '📚 שיעור הבא' }, { text: '🧪 מעבדה' }],
      [{ text: '🎯 אימון' }, { text: '📊 התקדמות' }],
      [{ text: '📋 מדריך מהיר' }, { text: '📚 תבניות' }],
      [{ text: '📖 מדריך טלגרם' }, { text: '❓ עזרה' }]
    ],
    resize_keyboard: true,
    one_time_keyboard: false
  };
}

// שימוש:
await this.bot.sendMessage(chatId, 'ברוכים הבאים!', {
  reply_markup: this.getMainKeyboard()
});

// טיפול בלחיצות:
if (text === '📚 שיעור הבא') {
  await cmdHandler.handleNext(msg);
} else if (text === '🧪 מעבדה') {
  await cmdHandler.handleSandbox(msg);
}
```

---

## סיכום

ספרייה זו מכילה תבניות קוד מעשיות ומוכנות לשימוש למפתחי בוטים בטלגרם. כל תבנית כוללת:
- **הסבר קצר** – למה זה שימושי
- **קוד מוכן** – ניתן להעתקה והדבקה
- **דוגמאות שימוש** – איך להשתמש בפועל

התבניות מבוססות על קוד אמיתי מפרויקט Markdown Trainer Bot ומוכחות בשטח.

---

**נוצר על ידי:** Markdown Trainer Bot Team  
**מקור:** [GitHub Repository](https://github.com/yourusername/markdown-trainer-bot)
