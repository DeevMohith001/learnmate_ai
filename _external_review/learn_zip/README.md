# LearnMate AI

LearnMate AI is now organized as a cleaner multi-page Streamlit project with the same core features as before, powered by Ollama instead of `llama-cpp-python`.

## Pages
- `app.py`: overview and quick start
- `pages/1_Operations.py`: create users, log study sessions, store quiz results
- `pages/2_Analytics.py`: insights, charts, recommendations, Spark summaries
- `pages/3_AI_Workspace.py`: summarizer, adaptive quiz generator, tutor chat
- `pages/4_System.py`: environment and database health view

## Core Features
- user creation
- study session logging
- quiz result storage
- event logging
- learning analytics
- knowledge score tracking
- AI recommendations
- AI summarization
- AI quiz generation
- AI tutor chat
- optional Spark-based summaries

## Tech Stack
- Streamlit
- SQLAlchemy with SQLite by default
- pandas
- Ollama for local AI
- optional PySpark
- python-dotenv

## Setup
1. Install Ollama from https://ollama.com
2. Pull a model:

```bash
ollama pull phi3:mini
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` if you want custom settings
5. Run the app:

```bash
streamlit run app.py
```

## Environment Variables
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
DATABASE_URL=sqlite:///learnmate_ai.db
```

## Notes
- SQLite is the default database so the project works immediately.
- You can switch to MySQL later by setting `DATABASE_URL` to a MySQL SQLAlchemy connection string.
- If Ollama is not running, AI tools will return a clear error message.
