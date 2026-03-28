from collections import defaultdict
from threading import Lock
from typing import Any, DefaultDict, Dict, List

from app.schemas import ChatMessage


class ConversationMemory:
    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self._store: DefaultDict[str, List[ChatMessage]] = defaultdict(list)
        self._context: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> List[ChatMessage]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def append(self, session_id: str, message: ChatMessage) -> List[ChatMessage]:
        with self._lock:
            messages = self._store[session_id]
            messages.append(message)
            if len(messages) > self.max_messages:
                del messages[:-self.max_messages]
            return list(messages)

    def get_context(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._context.get(session_id, {}))

    def set_context(self, session_id: str, context: Dict[str, Any]) -> None:
        with self._lock:
            self._context[session_id] = dict(context)


memory_store = ConversationMemory()
