// Wait for DOM to be fully loaded
(function() {
  'use strict';
  
  // Prevent this script from being copied
  const scriptTag = document.currentScript;
  if (scriptTag) {
    scriptTag.setAttribute('data-no-copy', 'true');
  }
  
  // Make copyCode available globally immediately (before DOMContentLoaded)
  window.copyCode = function(button) {
    try {
      // Find the parent snippet-content container
      const snippetContent = button.closest('.snippet-content');
      if (!snippetContent) {
        console.error('Could not find snippet-content');
        return;
      }
      
      // Find the pre element - must be a direct child of snippet-content
      // and must NOT be inside a script tag
      let preElement = null;
      
      // First try: next sibling of button (most common case)
      let sibling = button.nextElementSibling;
      if (sibling && sibling.tagName === 'PRE') {
        // Verify it's not inside a script
        if (!sibling.closest('script') && !sibling.closest('[data-no-copy="true"]')) {
          preElement = sibling;
        }
      }
      
      // Second try: find first PRE that is a direct child of snippet-content
      if (!preElement) {
        const children = Array.from(snippetContent.children);
        for (let i = 0; i < children.length; i++) {
          const child = children[i];
          if (child.tagName === 'PRE') {
            // Verify it's not inside a script and is a direct child
            if (!child.closest('script') && 
                !child.closest('[data-no-copy="true"]') &&
                child.parentElement === snippetContent) {
              preElement = child;
              break;
            }
          }
        }
      }
      
      if (!preElement || preElement.tagName !== 'PRE') {
        console.error('Could not find valid pre element in snippet-content');
        return;
      }
      
      // Final verification - make sure pre is not inside script
      if (preElement.closest('script') || preElement.closest('[data-no-copy="true"]')) {
        console.error('Pre element is inside script tag or marked as no-copy - skipping');
        return;
      }
      
      // Find the code element inside the pre
      const code = preElement.querySelector('code');
      if (!code) {
        console.error('Could not find code element');
        return;
      }
      
      // Final verification for code element
      if (code.closest('script') || code.closest('[data-no-copy="true"]')) {
        console.error('Code element is inside script tag or marked as no-copy - skipping');
        return;
      }
      
      // Get text content - use cloneNode to avoid affecting the original
      const codeClone = code.cloneNode(true);
      
      // Remove all span elements (Prism.js syntax highlighting)
      const spans = codeClone.querySelectorAll('span');
      spans.forEach(span => {
        const parent = span.parentNode;
        if (parent) {
          while (span.firstChild) {
            parent.insertBefore(span.firstChild, span);
          }
          parent.removeChild(span);
        }
      });
      
      // Get clean text content
      let text = codeClone.textContent || codeClone.innerText || '';
      
      // Additional cleanup - remove any script-related content that might have leaked
      // But be careful not to remove legitimate code that contains these patterns
      // Only remove if it looks like actual script tags
      // Note: We split the regex pattern to avoid </script> appearing in the code
      const scriptEnd = '</' + 'script>';
      const scriptTagPattern = new RegExp('<script[^>]*>[\\s\\S]*?' + scriptEnd, 'gi');
      text = text.replace(scriptTagPattern, '');
      text = text.replace(/&lt;script[^&]*&gt;[\s\S]*?&lt;\/script&gt;/gi, '');
      text = text.trim();
      
      // Validate that we have meaningful content (at least 10 characters)
      if (!text || text.length < 10) {
        console.error('No valid text to copy or text too short');
        return;
      }
      
      // Additional safety check - if text contains our script function signature, skip it
      if (text.includes('window.copyCode = function') || 
          text.includes('document.addEventListener(\'DOMContentLoaded\'')) {
        console.error('Detected script code in text - skipping copy');
        return;
      }
      
      // Copy to clipboard
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
          // Visual feedback
          const originalText = button.textContent;
          button.textContent = '✓ הועתק!';
          button.classList.add('copy-success');
          
          setTimeout(() => {
            button.textContent = originalText;
            button.classList.remove('copy-success');
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy:', err);
          fallbackCopyText(text, button);
        });
      } else {
        fallbackCopyText(text, button);
      }
    } catch (err) {
      console.error('Error in copyCode:', err);
      button.textContent = '❌ שגיאה';
      setTimeout(() => {
        button.textContent = 'העתק';
      }, 2000);
    }
  };
  
  // Fallback copy function
  function fallbackCopyText(text, button) {
    try {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.top = '0';
      textArea.style.left = '0';
      textArea.style.width = '2em';
      textArea.style.height = '2em';
      textArea.style.padding = '0';
      textArea.style.border = 'none';
      textArea.style.outline = 'none';
      textArea.style.boxShadow = 'none';
      textArea.style.background = 'transparent';
      textArea.style.opacity = '0';
      textArea.style.zIndex = '-9999';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      
      try {
        const successful = document.execCommand('copy');
        if (successful) {
          const originalText = button.textContent;
          button.textContent = '✓ הועתק!';
          button.classList.add('copy-success');
          
          setTimeout(() => {
            button.textContent = originalText;
            button.classList.remove('copy-success');
          }, 2000);
        } else {
          button.textContent = '❌ שגיאה';
          setTimeout(() => {
            button.textContent = 'העתק';
          }, 2000);
        }
      } catch (err) {
        console.error('execCommand failed:', err);
        button.textContent = '❌ שגיאה';
        setTimeout(() => {
          button.textContent = 'העתק';
        }, 2000);
      }
      
      document.body.removeChild(textArea);
    } catch (err) {
      console.error('Fallback copy failed:', err);
      button.textContent = '❌ שגיאה';
      setTimeout(() => {
        button.textContent = 'העתק';
      }, 2000);
    }
  }
  
  // Wait for DOM to be fully loaded for other functionality
  document.addEventListener('DOMContentLoaded', function() {
    // Search functionality
    const searchInput = document.getElementById('searchInput');
    const snippets = document.querySelectorAll('.snippet');
    const categories = document.querySelectorAll('.category');
    const noResults = document.getElementById('noResults');

    if (searchInput) {
      searchInput.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase().trim();
        let visibleCount = 0;

        if (searchTerm === '') {
          // Show all
          snippets.forEach(snippet => {
            snippet.classList.remove('filtered-out');
          });
          categories.forEach(category => {
            category.classList.remove('no-matches');
          });
          if (noResults) {
            noResults.classList.remove('show');
          }
          return;
        }

        // Search in snippets
        snippets.forEach(snippet => {
          const title = snippet.querySelector('.snippet-title');
          const description = snippet.querySelector('.snippet-description');
          const code = snippet.querySelector('code');
          
          const titleText = title ? title.textContent.toLowerCase() : '';
          const descText = description ? description.textContent.toLowerCase() : '';
          let codeText = '';
          if (code) {
            // Get text without HTML tags
            const codeClone = code.cloneNode(true);
            const spans = codeClone.querySelectorAll('span');
            spans.forEach(span => {
              const parent = span.parentNode;
              if (parent) {
                while (span.firstChild) {
                  parent.insertBefore(span.firstChild, span);
                }
                parent.removeChild(span);
              }
            });
            codeText = (codeClone.textContent || codeClone.innerText || '').toLowerCase();
          }
          
          const matches = titleText.includes(searchTerm) || 
                        descText.includes(searchTerm) || 
                        codeText.includes(searchTerm);
          
          if (matches) {
            snippet.classList.remove('filtered-out');
            visibleCount++;
          } else {
            snippet.classList.add('filtered-out');
          }
        });

        // Hide categories with no visible snippets
        categories.forEach(category => {
          const visibleSnippets = category.querySelectorAll('.snippet:not(.filtered-out)');
          if (visibleSnippets.length === 0) {
            category.classList.add('no-matches');
          } else {
            category.classList.remove('no-matches');
          }
        });

        // Show/hide no results message
        if (noResults) {
          if (visibleCount === 0) {
            noResults.classList.add('show');
          } else {
            noResults.classList.remove('show');
          }
        }
      });
    }

    // Back to top button
    const backToTopButton = document.getElementById('backToTop');
    
    if (backToTopButton) {
      // Show/hide button based on scroll position
      function handleScroll() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
        if (scrollTop > 300) {
          backToTopButton.classList.add('show');
        } else {
          backToTopButton.classList.remove('show');
        }
      }

      // Use throttled scroll handler for better performance
      let scrollTimeout;
      window.addEventListener('scroll', function() {
        if (scrollTimeout) {
          clearTimeout(scrollTimeout);
        }
        scrollTimeout = setTimeout(handleScroll, 10);
      }, { passive: true });
      
      // Check initial scroll position
      handleScroll();

      // Smooth scroll to top
      backToTopButton.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        window.scrollTo({
          top: 0,
          behavior: 'smooth'
        });
      });
    }

    // Initialize Prism.js highlighting
    if (typeof Prism !== 'undefined') {
      Prism.highlightAll();
    }
  });
})();
