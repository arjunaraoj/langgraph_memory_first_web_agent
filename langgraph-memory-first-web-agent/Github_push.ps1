# 1. Go to D drive
cd D:\

# 2. Create temporary GitHub upload folder
mkdir github-upload
cd github-upload

# 3. Clone your GitHub repo
git clone https://github.com/arjunaraoj/memory_first_web_agent.git

# 4. Go inside cloned repo
cd memory_first_web_agent

# 5. Create folder inside GitHub repo
mkdir langgraph-memory-first-web-agent

# 6. Copy your project files into that folder
robocopy D:\langgraph-memory-first-web-agent .\langgraph-memory-first-web-agent /E /XD venv .venv __pycache__ .ipynb_checkpoints .pytest_cache logs .git /XF .env *.pyc *.log

# 7. Create .gitignore at repo root
@"
.env
*.env

venv/
.venv/

__pycache__/
*.pyc
*.pyo
*.pyd

.ipynb_checkpoints/
.pytest_cache/

logs/
*.log

.DS_Store
Thumbs.db

.vscode/
.idea/
"@ | Set-Content .gitignore

# 8. Check status
git status

# 9. Add files
git add .

# 10. Commit
git commit -m "Add langgraph memory first web agent project folder"

# 11. Push to GitHub
git push origin main