from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:Na999123hA@127.0.0.1:5432/custumers"
engine = create_engine(DATABASE_URL)

class Customer:
    def __init__(self, name: str, phone: str, address: str):
        self.name = name
        self.phone = phone
        self.address = address

    # 1. Принимаем user_id в метод save
    def save(self, user_id: str) -> None:
        # Добавляем user_id в INSERT
        add_customer_query = text("""
            INSERT INTO users (name, phone, address, user_id)
            VALUES (:name, :phone, :address, :user_id)
        """)

        try:
            with engine.connect() as conn:
                conn.execute(
                    add_customer_query,
                    {
                        "name": self.name,
                        "phone": self.phone,
                        "address": self.address,
                        "user_id": user_id  # Передаем в запрос
                    }
                )
                conn.commit()
            print("Данные успешно добавлены в БД")
            
        except Exception as e:
            print("Произошла ошибка при работе с БД:")
            print(e)
            raise e # Пробрасываем ошибку выше, чтобы узел LangGraph её залогировал
            
    # 2. Исправляем проверку: ищем именно по колонке user_id, а не просто id
    @staticmethod
    def get_by_id(user_id: str):
        find_query = text("""
            SELECT name, phone, address FROM users WHERE user_id = :user_id 
        """)
        with engine.connect() as conn:
            result = conn.execute(find_query, {"user_id": user_id}).fetchone()
            return {"name": result[0], "phone": result[1], "address": result[2]} if result else None