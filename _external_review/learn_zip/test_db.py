from database.db_connection import init_database
from database.queries import create_user, get_users_df


if __name__ == "__main__":
    init_database()
    if get_users_df().empty:
        create_user("Demo User", "demo@example.com")
    print("Database initialized and ready.")
