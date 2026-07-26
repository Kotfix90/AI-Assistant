from typing import Annotated, Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# --- Профиль клиента ---
class ClientProfile(BaseModel):
    status: Literal["guest", "registering", "already_registered"] = "guest"
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

    def is_complete(self) -> bool:
        return bool(self.name and self.phone and self.address)


# --- Предпочтения клиента (Память подбора) ---
class UserPreferences(BaseModel):
    area_sqm: Optional[float] = Field(None, description="Площадь помещения в кв. м")
    budget_max: Optional[int] = Field(None, description="Максимальный бюджет в рублях")
    preferred_brands: List[str] = Field(default_factory=list, description="Предпочитаемые бренды")


# --- Основной State графа ---
class AgentState(TypedDict, total=False):
    messages: Annotated[List[Any], add_messages]
    client_info: Dict[str, Any]       # Храним как dict для гарантированной совместимости с JSON
    user_preferences: Dict[str, Any]  # Память характеристик (площадь, бюджет и т.д.)
    user_id: str
    next_step: Optional[str]          # Флаг текущего шага FSM ("ask_name", "ask_phone", "ask_address", "checkout")


# --- Схемы извлечения данных (Structured Output) ---
class ExtractName(BaseModel):
    name: Optional[str] = Field(
        default=None, 
        description="Имя человека. Например: 'Олег', 'Иван'. Если имени нет в тексте, верни null."
    )

class ExtractPhone(BaseModel):
    phone: Optional[str] = Field(
        default=None, 
        description="Номер телефона (например '89991112233', '+79991112233'). Если телефона нет, верни null."
    )

class ExtractAddress(BaseModel):
    address: Optional[str] = Field(
        default=None, 
        description="Адрес доставки или монтажа (например 'Челябинск, ул. Ленина 10, кв 5'). Если адреса нет, верни null."
    )

class IntentDecision(BaseModel):
    intent: Literal["rag", "about_company", "register", "off_topic"] = Field(
        description="Намерение пользователя с учетом контекста беседы."
    )