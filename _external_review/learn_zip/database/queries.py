import pandas as pd
from sqlalchemy import select

from database.db_connection import SessionLocal
from database.models import Event, QuizResult, StudySession, User


def create_user(name, email):
    clean_name = name.strip()
    clean_email = email.strip().lower()
    with SessionLocal() as session:
        existing = session.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        if existing is not None:
            return existing.id

        user = User(name=clean_name, email=clean_email)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.id


def log_study_session(user_id, subject, topic, time_spent):
    with SessionLocal() as session:
        record = StudySession(
            user_id=user_id,
            subject=subject.strip(),
            topic=topic.strip(),
            time_spent=int(time_spent),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def save_quiz_result(user_id, subject, topic, score, total_questions):
    with SessionLocal() as session:
        record = QuizResult(
            user_id=user_id,
            subject=subject.strip(),
            topic=topic.strip(),
            score=float(score),
            total_questions=int(total_questions),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def log_event(user_id, event_type, event_data):
    with SessionLocal() as session:
        event = Event(user_id=user_id, event_type=event_type, event_data=event_data)
        session.add(event)
        session.commit()
        session.refresh(event)
        return event.id


def get_users_df():
    with SessionLocal() as session:
        rows = session.execute(select(User).order_by(User.id.desc())).scalars().all()
    return pd.DataFrame([
        {"id": row.id, "name": row.name, "email": row.email, "created_at": row.created_at}
        for row in rows
    ])


def get_study_df():
    with SessionLocal() as session:
        rows = session.execute(select(StudySession).order_by(StudySession.id.desc())).scalars().all()
    return pd.DataFrame([
        {
            "id": row.id,
            "user_id": row.user_id,
            "subject": row.subject,
            "topic": row.topic,
            "time_spent": row.time_spent,
            "created_at": row.created_at,
        }
        for row in rows
    ])


def get_quiz_df():
    with SessionLocal() as session:
        rows = session.execute(select(QuizResult).order_by(QuizResult.id.desc())).scalars().all()
    data = []
    for row in rows:
        percent = (row.score / row.total_questions) * 100 if row.total_questions else 0
        data.append(
            {
                "id": row.id,
                "user_id": row.user_id,
                "subject": row.subject,
                "topic": row.topic,
                "score": row.score,
                "total_questions": row.total_questions,
                "score_percent": round(percent, 2),
                "created_at": row.created_at,
            }
        )
    return pd.DataFrame(data)


def get_events_df(limit=100):
    with SessionLocal() as session:
        rows = session.execute(select(Event).order_by(Event.id.desc()).limit(limit)).scalars().all()
    return pd.DataFrame([
        {
            "id": row.id,
            "user_id": row.user_id,
            "event_type": row.event_type,
            "event_data": row.event_data,
            "created_at": row.created_at,
        }
        for row in rows
    ])
