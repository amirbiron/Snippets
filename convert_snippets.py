#!/usr/bin/env python3
"""
Convert Markdown snippets to HTML format for index.html

Usage:
    python convert_snippets.py <input.md> <output.html> --offset <number>

Example:
    python convert_snippets.py Snippets5New.md snippets_17_28.html --offset 16
"""

import re
import sys
import argparse

# Default icon mappings for categories
DEFAULT_CATEGORY_ICONS = {
    # Snippets5New categories
    "Webhook Handling & Security": "shield-alt",
    "Rate Limiting מתקדם": "tachometer-alt",
    "Retry & Circuit Breaker Patterns": "redo",
    "Telegram Inline Queries": "search",
    "Message Scheduling & Delayed Tasks": "clock",
    "Database Migrations": "database",
    "API Client Patterns": "plug",
    "Decorators שימושיים": "magic",
    "Telegram Payments & Invoices": "credit-card",
    "Monitoring & Metrics": "chart-line",
    "Environment & Secrets Management": "key",
    "Data Serialization & Validation": "check-circle",

    # Snippets6New categories
    "Conversation Handlers & State Machines": "comments",
    "Telegram Media Handling": "photo-video",
    "Internationalization (i18n)": "globe",
    "Command Permissions & Access Control": "lock",
    "Graceful Shutdown & Cleanup": "power-off",
    "Batch Processing & Bulk Operations": "layer-group",
    "Telegram Channel/Group Management": "users",
    "Data Export/Import": "file-export",
    "Performance Monitoring": "chart-bar",
    "Webhook vs Polling Patterns": "exchange-alt",
    "Telegram Bot Commands Menu": "bars",
    "Advanced Error Recovery": "life-ring",
}


def escape_html(text):
    """Escape HTML special characters in code blocks"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def parse_md_to_html(md_content, category_offset=0, icon_map=None):
    """
    Convert markdown snippets to HTML format

    Args:
        md_content: Markdown file content
        category_offset: Number to add to category numbers (e.g., 16 for categories 17+)
        icon_map: Dictionary mapping category titles to FontAwesome icon names

    Returns:
        HTML string
    """
    if icon_map is None:
        icon_map = DEFAULT_CATEGORY_ICONS

    html_parts = []

    # Split by main categories (## 1., ## 2., etc.)
    category_sections = re.split(r'^## (\d+)\. (.+?)$', md_content, flags=re.MULTILINE)

    # Skip first element (header before first category)
    for i in range(1, len(category_sections), 3):
        if i + 2 >= len(category_sections):
            break

        cat_num_in_file = category_sections[i]
        cat_title = category_sections[i + 1].strip()
        cat_content = category_sections[i + 2]

        # Calculate HTML category number
        html_cat_num = int(cat_num_in_file) + category_offset
        icon = icon_map.get(cat_title, "code")

        html_parts.append(f'''
      <!-- Category {html_cat_num}: {cat_title} -->
      <div class="category" id="category-{html_cat_num}">
        <div class="category-header">
          <i class="fas fa-{icon}"></i>
          <h2>{cat_title}</h2>
        </div>
''')

        # Find all snippets within this category
        # Pattern supports both "מטרה:" and "למה זה שימושי:"
        snippet_pattern = r'###\s+([^\n]+)\n\*\*(?:מטרה|למה זה שימושי):\*\*\s+([^\n]+)\n\s*```python\n(.*?)```'
        snippets = re.findall(snippet_pattern, cat_content, re.DOTALL)

        for snippet_idx, (snippet_title, snippet_desc, code) in enumerate(snippets, 1):
            snippet_id = f"{html_cat_num}-{snippet_idx}"

            # Determine description prefix based on what's in the original
            desc_prefix = "מטרה:" if "**מטרה:**" in cat_content else "למה זה שימושי:"

            html_parts.append(f'''
        <div class="snippet" id="snippet-{snippet_id}">
          <div class="snippet-header">
            <h3 class="snippet-title">{html_cat_num}.{snippet_idx} {snippet_title.strip()}</h3>
            <p class="snippet-description"><strong>{desc_prefix}</strong> {snippet_desc.strip()}</p>
          </div>
          <div class="snippet-content">
            <button class="copy-button" onclick="copyCode(this)">העתק</button>
            <pre class="language-python"><code class="language-python">{escape_html(code.strip())}</code></pre>
          </div>
        </div>
''')

        html_parts.append('      </div>\n')

    return ''.join(html_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Markdown snippets to HTML format for index.html'
    )
    parser.add_argument('input', help='Input markdown file (e.g., Snippets5New.md)')
    parser.add_argument('output', help='Output HTML file (e.g., snippets_17_28.html)')
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Category number offset (e.g., 16 to start at category 17)'
    )

    args = parser.parse_args()

    # Read markdown file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{args.input}' not found")
        sys.exit(1)

    # Convert to HTML
    html_output = parse_md_to_html(md_content, category_offset=args.offset)

    # Write output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_output)

    # Print statistics
    cat_count = len(re.findall('<div class="category"', html_output))
    snippet_count = len(re.findall('<div class="snippet"', html_output))

    print(f"✅ Conversion complete!")
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Categories: {cat_count}")
    print(f"   Snippets: {snippet_count}")
    print(f"   Size: {len(html_output)} bytes")


if __name__ == '__main__':
    main()
