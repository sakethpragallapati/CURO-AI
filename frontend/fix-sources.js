const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'components', 'ChatInterface.tsx');
let content = fs.readFileSync(filePath, 'utf-8');

// Fix 1: Replace the hostname extraction + return block with URL-validated version
const oldHostname = `let hostname = '';
                                              try { hostname = new URL(src.url).hostname.replace('www.', ''); } catch { hostname = src.url; }
                                              return (`;

const newHostname = `let hostname = '';
                                              let isValidUrl = false;
                                              try { const u = new URL(src.url); hostname = u.hostname.replace('www.', ''); isValidUrl = u.protocol === 'https:' || u.protocol === 'http:'; } catch { hostname = ''; isValidUrl = false; }
                                              if (!isValidUrl) return null;
                                              return (`;

if (content.includes(oldHostname)) {
  content = content.replace(oldHostname, newHostname);
  console.log('[OK] Fix 1: URL validation added');
} else {
  console.log('[SKIP] Fix 1: Pattern not found (may already be applied)');
}

// Fix 2: Add cursor-pointer to the source card <a> tag
const oldClass = `hover:shadow-curo-teal/5"`;
const newClass = `hover:shadow-curo-teal/5 cursor-pointer"`;

// Only replace in the source cards section (check it doesn't already have cursor-pointer)
if (content.includes(oldClass) && !content.includes(newClass)) {
  content = content.replace(oldClass, newClass);
  console.log('[OK] Fix 2: cursor-pointer added');
} else {
  console.log('[SKIP] Fix 2: Already applied or not found');
}

fs.writeFileSync(filePath, content, 'utf-8');
console.log('[DONE] File updated successfully');
