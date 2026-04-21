const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'components', 'ChatInterface.tsx');
const content = fs.readFileSync(filePath, 'utf-8');
const lines = content.split('\n');

// Find the target lines
let hostnameLineIdx = -1;
let classNameLineIdx = -1;

for (let i = 0; i < lines.length; i++) {
  const trimmed = lines[i].trim();
  if (trimmed.includes("hostname = new URL(src.url).hostname.replace") && trimmed.includes("hostname = src.url")) {
    hostnameLineIdx = i;
  }
  if (trimmed.includes("hover:shadow-curo-teal/5") && !trimmed.includes("cursor-pointer") && trimmed.includes("group block")) {
    classNameLineIdx = i;
  }
}

console.log('Hostname line index:', hostnameLineIdx);
console.log('ClassName line index:', classNameLineIdx);

if (hostnameLineIdx >= 0) {
  // Get the leading whitespace
  const ws = lines[hostnameLineIdx].match(/^(\s*)/)[1];
  
  // Replace the single hostname line with validation logic (3 lines)  
  const newLines = [
    ws + "let isValidUrl = false;",
    ws + "try { const u = new URL(src.url); hostname = u.hostname.replace('www.', ''); isValidUrl = u.protocol === 'https:' || u.protocol === 'http:'; } catch { hostname = ''; isValidUrl = false; }",
    ws + "if (!isValidUrl) return null;"
  ];
  
  lines.splice(hostnameLineIdx, 1, ...newLines);
  console.log('Replaced hostname line with URL validation');
  
  // Adjust classNameLineIdx if it was after hostnameLineIdx
  if (classNameLineIdx > hostnameLineIdx) {
    classNameLineIdx += 2; // We added 2 extra lines
  }
}

if (classNameLineIdx >= 0) {
  lines[classNameLineIdx] = lines[classNameLineIdx].replace(
    'hover:shadow-curo-teal/5"',
    'hover:shadow-curo-teal/5 cursor-pointer"'
  );
  console.log('Added cursor-pointer to source card');
}

fs.writeFileSync(filePath, lines.join('\n'), 'utf-8');
console.log('File saved successfully!');

// Verify
const verify = fs.readFileSync(filePath, 'utf-8');
console.log('Contains isValidUrl:', verify.includes('isValidUrl'));
console.log('Contains cursor-pointer in source card:', verify.includes('cursor-pointer'));
