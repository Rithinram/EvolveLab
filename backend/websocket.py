"""
EvolveLab — WebSocket Manager
Handles real-time event broadcasting to connected clients.
"""

import json
import logging
import asyncio
from typing import List, Set
from fastapi import WebSocket

logger = logging.getLogger("evolvelab.websocket")


class WebSocketManager:
    """Manages WebSocket connections and broadcasts evolution events."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected (%d total)", len(self.active_connections))

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        text = json.dumps(message, default=str)
        disconnected = []

        for ws in self.active_connections:
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    def sync_broadcast(self, message: dict):
        """Synchronous broadcast helper for use from evolution thread."""
        if self.loop is None or not self.loop.is_running():
            return
            
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
        except Exception as e:
            logger.error(f"WS Sync Broadcast Error: {e}")
