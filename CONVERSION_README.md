# Snippet Conversion Tool

כלי להמרת קבצי Markdown של snippets לפורמט HTML להוספה ל-index.html.

## שימוש מהיר

### המרת קובץ Markdown ל-HTML

```bash
python3 convert_snippets.py <input.md> <output.html> --offset <number>
```

**דוגמאות:**

```bash
# להמרת Snippets5New.md (קטגוריות 1-12 → 17-28)
python3 convert_snippets.py Snippets5New.md snippets_17_28.html --offset 16

# להמרת Snippets6New.md (קטגוריות 13-24 → 29-40)
python3 convert_snippets.py Snippets6New.md snippets_29_40.html --offset 16

# להמרת Snippets7New.md (קטגוריות 1-10 → 41-50)
python3 convert_snippets.py Snippets7New.md snippets_41_50.html --offset 40
```

### הוספה ל-index.html

לאחר ההמרה, הוסף את התוכן ל-index.html:

```bash
python3 << 'PYEOF'
# Read both files
with open('index.html', 'r', encoding='utf-8') as f:
    index_html = f.read()

with open('snippets_17_28.html', 'r', encoding='utf-8') as f:
    new_categories = f.read()

# Find insertion point (before closing main container)
insertion_marker = '    </div>\n\n    <!-- Back to top button -->'

if insertion_marker in index_html:
    updated_html = index_html.replace(
        insertion_marker,
        f'{new_categories}    </div>\n\n    <!-- Back to top button -->'
    )

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(updated_html)

    print("✅ Successfully added categories to index.html")
else:
    print("❌ Could not find insertion point")
PYEOF
```

## איך לחשב את ה-offset?

ה-offset הוא המספר שנוסיף למספרי הקטגוריות בקובץ ה-Markdown.

**נוסחה:**
```
offset = (מספר הקטגוריה האחרונה ב-index.html) - (מספר הקטגוריה הראשונה בקובץ ה-Markdown) + 1
```

**דוגמאות:**

| index.html יש | הקובץ מתחיל מ- | offset | תוצאה |
|---------------|----------------|--------|-------|
| קטגוריות 1-16 | ## 1.          | 16     | 17-28 |
| קטגוריות 1-28 | ## 13.         | 16     | 29-40 |
| קטגוריות 1-40 | ## 1.          | 40     | 41-50 |

## פורמט קובץ Markdown

הקובץ צריך להיות בפורמט הזה:

```markdown
# Title

## 1. Category Name

### Snippet Title
**מטרה:** תיאור מה הסניפט עושה

​```python
# קוד פייתון כאן
def example():
    pass
​```

### Snippet Title 2
**למה זה שימושי:** תיאור נוסף

​```python
# עוד קוד
​```

## 2. Another Category

...
```

**שים לב:**
- הכלי תומך גם ב-"מטרה:" וגם ב-"למה זה שימושי:"
- חייב רווח אחרי `###` ואחרי `##`
- בלוק הקוד חייב להתחיל ב-\`\`\`python

## הוספת אייקונים לקטגוריות חדשות

אם יש קטגוריות חדשות, פתח את `convert_snippets.py` והוסף ל-`DEFAULT_CATEGORY_ICONS`:

```python
DEFAULT_CATEGORY_ICONS = {
    "שם הקטגוריה": "font-awesome-icon-name",
    # לדוגמה:
    "Database Operations": "database",
    "API Integration": "plug",
}
```

[רשימת אייקונים של Font Awesome](https://fontawesome.com/icons)

## פתרון בעיות

### "Found 0 snippets"

**בעיה:** הסקריפט לא מצא snippets בקובץ.

**פתרונות:**
1. ודא שהפורמט נכון (ראה למעלה)
2. בדוק שיש רווח אחרי `###`
3. ודא שבלוק הקוד מתחיל ב-\`\`\`python (ולא \`\`\` בלי python)
4. בדוק שיש **מטרה:** או **למה זה שימושי:** לפני בלוק הקוד

### "Could not find insertion point"

**בעיה:** לא מצא איפה להכניס את ה-HTML ב-index.html.

**פתרון:** בדוק שיש את השורות האלה ב-index.html:
```html
    </div>

    <!-- Back to top button -->
```

## דוגמת תהליך מלא

```bash
# 1. המר markdown ל-HTML
python3 convert_snippets.py Snippets7New.md snippets_new.html --offset 40

# 2. הוסף ל-index.html
python3 << 'PYEOF'
with open('index.html', 'r', encoding='utf-8') as f:
    index_html = f.read()
with open('snippets_new.html', 'r', encoding='utf-8') as f:
    new_categories = f.read()
insertion_marker = '    </div>\n\n    <!-- Back to top button -->'
updated_html = index_html.replace(insertion_marker, f'{new_categories}    </div>\n\n    <!-- Back to top button -->')
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(updated_html)
print("✅ Done!")
PYEOF

# 3. בדוק שהכל עובד
# פתח את index.html בדפדפן

# 4. נקה קבצים זמניים ועשה commit
rm snippets_new.html
git add index.html convert_snippets.py CONVERSION_README.md
git commit -m "Add new snippets to index.html"
git push
```

## תחזוקה

הסקריפט הזה שמור ב-repository כדי ש:
1. תוכל להשתמש בו בעתיד בלי לכתוב אותו מחדש
2. AI אחרים יוכלו למצוא אותו ולהשתמש בו
3. יש תיעוד ברור איך להשתמש בו

אם יש שינויים בפורמט ה-HTML של index.html או בפורמט ה-Markdown, עדכן את הסקריפט בהתאם.
