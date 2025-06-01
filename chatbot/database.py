from dotenv import load_dotenv
import os
import psycopg2
import hashlib

class Database:
    def __init__(self):
        load_dotenv()

    def hash_password(self, password: str) -> str:
        """
        Create a SHA-256 hash of the password.
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def get_db_connection(self):
        """
        Establish a connection to Postgres using psycopg2 and env vars.
        """
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432"),
            sslmode=os.getenv("DB_SSLMODE", "require")
        )

    def authenticate(self, key) -> dict:
        """
        Simple API key check for saving queries.
        """
        api_key = key
        if not api_key or api_key != os.getenv('MASTER_KEY'):
            return {'error': 'Forbidden: Invalid API Key', 'status': 403}
        return {'status': 200}

    def save_user(self, user_id: str, password: str, key) -> dict:
        """
        Register a new user; returns status dict.
        """
        password_hash = self.hash_password(password)

        auth = self.authenticate(key)

        if auth['status'] != 200:
            return auth

        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM users WHERE user_id = %s", 
                        (user_id,)
                    )
                    if cursor.fetchone():
                        return {'error': 'User already exists', 'status': 409}
                    cursor.execute(
                        "INSERT INTO users (user_id, password_hash, created_at) VALUES (%s, %s, CURRENT_TIMESTAMP)",
                        (user_id, password_hash)
                    )
                    
            return {'status': 200, 'message': 'User created successfully'}

        except Exception as e:
            return {'error': 'Error creating user', 'details': str(e), 'status': 500}

    def login_user(self, user_id: str, password: str, key) -> dict:
        """
        Authenticate existing user; returns status dict.
        """
        password_hash = self.hash_password(password)

        auth = self.authenticate(key)
        
        if auth['status'] != 200:
            return auth

        try:
            result = None
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                    "SELECT password_hash FROM users WHERE user_id = %s",
                    (user_id,)
                    )
                    result = cursor.fetchone()

            if not result:
                return {'error': 'User not found', 'status': 404}

            stored_hash = result[0]
            if stored_hash == password_hash:
                return {'status': 200, 'message': 'Login successful', 'user': user_id}
            else:
                return {'error': 'Invalid credentials', 'status': 401}

        except Exception as e:
            return {'error': 'Error during login', 'details': str(e), 'status': 500}

    def upload_messages(self, user, question, answer, video, chat_id, key) -> dict:
        """
        Insert into questions table if API key valid.
        """
        
        auth = self.authenticate(key)
        if auth['status'] != 200:
            return auth
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO questions (
                            user_id, question, answer, video, chat_id, created_at, modified_at
                        ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (question) DO UPDATE 
                          SET answer = EXCLUDED.answer, modified_at = CURRENT_TIMESTAMP
                        """,
                        (user, question, answer, video, chat_id)
                    )
            
            return {'status': 200, 'message': 'Query uploaded successfully'}

        except Exception as e:
            return {'error': 'Error uploading query', 'details': str(e), 'status': 500}
        
    def delete_messages(self, user: str, chat: str, key: str) -> dict:
        """
        Deletes a chat
        """
        auth = self.authenticate(key)
        if auth['status'] != 200:
            return auth
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE 
                        FROM questions
                        WHERE chat_id = %s AND user_id = %s
                        """,
                        (chat, user)
                    )
            return {'message': 'succesfully deleted', 'status': 200}
        except Exception as e:
            return {'error': str(e), 'status': 500}

    def get_messages(self, user: str, key: str) -> dict:
        """
        Retrieve all Q&A pairs for a given user and conversation.
        Returns a list of { 'chat': chat_id, 'question': ..., 'answer': ..., 'video': ... }.
        """
        
        auth = self.authenticate(key)
        if auth['status'] != 200:
            return auth
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT question, answer, video, chat_id 
                        FROM questions 
                        WHERE user_id = %s 
                        ORDER BY created_at
                        """,
                        (user,)
                    )
                    rows = cursor.fetchall()
            
            conversation = [
                {'question': row[0], 'answer': row[1], 'video': row[2], 'chat_id': row[3]}
                for row in rows
            ]

            return {'status': 200, 'conversations': conversation}
        except Exception as e:
            return {'error': str(e), 'status': 500}