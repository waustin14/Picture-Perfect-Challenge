from typing import Dict, List
from fastapi import WebSocket
from asyncio import Lock


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = Lock()

    async def connect(self, game_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(game_id, []).append(websocket)

    async def disconnect(self, game_id: str, websocket: WebSocket):
        async with self._lock:
            conns = self._connections.get(game_id, [])
            if websocket in conns:
                conns.remove(websocket)

    async def broadcast(self, game_id: str, message: dict):
        async with self._lock:
            conns = list(self._connections.get(game_id, []))
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                # could log and drop dead sockets here
                pass
