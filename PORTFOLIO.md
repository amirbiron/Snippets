```yaml
name: "Snippets - אוסף קטעי קוד Python"
repo: "https://github.com/amirbiron/Snippets"
status: "פעיל"

one_liner: "אתר סטטי המציג אוסף מקיף של קטעי קוד Python מאורגנים בקטגוריות, עם כלי המרה מ-Markdown ל-HTML."

stack:
  - HTML5
  - CSS3
  - JavaScript (Vanilla)
  - Python (סקריפט המרה)
  - Font Awesome (אייקונים)

key_features:
  - "עשרות קטעי קוד Python מאורגנים בקטגוריות"
  - "ממשק מודרני ורספונסיבי עם חיפוש וניווט"
  - "העתקת קוד בלחיצה אחת (Copy to Clipboard)"
  - "כלי המרה אוטומטי מ-Markdown ל-HTML"
  - "תמיכה ב-offset לנומרציה רציפה של קטגוריות"
  - "אייקונים מותאמים לכל קטגוריה (Font Awesome)"

architecture:
  summary: |
    אתר סטטי (HTML/CSS/JS) ללא framework. קבצי Markdown מכילים את הסניפטים המקוריים,
    וסקריפט Python (convert_snippets.py) ממיר אותם לפורמט HTML להכנסה ל-index.html.
    scripts.js מנהל את הפונקציונליות האינטראקטיבית (העתקה, ניווט).
  entry_points:
    - "index.html - דף ראשי עם כל הסניפטים"
    - "scripts.js - לוגיקת צד לקוח (העתקה, ניווט)"
    - "convert_snippets.py - כלי המרה מ-Markdown ל-HTML"
    - "Snippets1-10.md - קבצי מקור של הסניפטים"

demo:
  live_url: "" # TODO: בדוק ידנית - אולי GitHub Pages
  video_url: "" # TODO: בדוק ידנית

setup:
  quickstart: |
    1. git clone <repository-url> && cd Snippets
    2. פתח index.html בדפדפן
    3. להוספת סניפטים: python3 convert_snippets.py <input.md> <output.html> --offset <N>

your_role: "פיתוח מלא - עיצוב האתר, כתיבת הסניפטים, בניית כלי ההמרה"

tradeoffs:
  - "אתר סטטי ללא framework - פשטות מקסימלית, ללא תלויות"
  - "קובץ HTML יחיד גדול - פשטות פריסה על חשבון זמני טעינה"
  - "סקריפט המרה ידני - שליטה מלאה בפורמט"

metrics: {} # TODO: בדוק ידנית

faq:
  - q: "איך מוסיפים סניפטים חדשים?"
    a: "כותבים קובץ Markdown, מריצים convert_snippets.py עם offset מתאים, ומוסיפים ל-index.html"
  - q: "מה ה-offset?"
    a: "מספר שנוסיף למספרי הקטגוריות ב-Markdown כדי להמשיך את הנומרציה הקיימת ב-index.html"
```
