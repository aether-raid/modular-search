# src/connection_manager.py

"""
Manages WebSocket connections and message buffering.

This class handles the lifecycle of WebSocket connections, including
accepting new connections, disconnecting clients, sending messages,
buffering messages for disconnected clients, and sending keepalive
pings at regular intervals.

Credits to Sik Feng.
"""

import asyncio
import json
import logging
import os
import signal
import argparse
import uvicorn
from typing import Any, Dict, List
from fastapi import WebSocket, WebSocketDisconnect, FastAPI
from .pipeline import pipeline_main

logger = logging.getLogger("ConnectionManager")


class ConnectionManager:
    """
    Manages WebSocket connections and message buffering.
    """
    END_OF_MESSAGE_RESPONSE = {"<END_OF_MESSAGE>": ""}
    KEEP_ALIVE_PING = {"<PING>": ""}

    def __init__(self) -> None:
        """
        Initializes the ConnectionManager with empty lists for active
        connections and message buffer.
        """
        self.active_connections: List[WebSocket] = []
        self.message_buffer: Dict[str, List[Dict[str, Any]]] = {}

    async def _on_connect(self, websocket: WebSocket, session_id: str) -> None:
        """
        Accepts a new WebSocket connection and adds it to the list of
        active connections.

        :param websocket: The WebSocket connection to accept.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connection accepted")

    async def _on_disconnect(self, websocket: WebSocket) -> None:
        """
        Removes a WebSocket connection from the list of active
        connections.

        :param websocket: The WebSocket connection to remove.
        """
        self.active_connections.remove(websocket)
        logger.info("Client disconnected")

    async def _on_receive(self, websocket: WebSocket,
                          session_id: str, data: Dict[str, Any]) -> None:
        """
        Processes incoming messages and performs the corresponding
        actions based on the method specified in the message.

        :param websocket: The WebSocket connection from which the
            message was received.
        :param session_id: The session identifier for the connection.
        :param data: The data received from the WebSocket connection.
            The expected format of the `data` parameter is a dictionary
            with at least two keys: `method` and `params`.

        methods:
            - query: Process a user query and return
                a list of related codebases.
            - shutdown: Shutdown the agent manager.

        """
        method = data.get("method")
        params = data.get("params", {})

        if method == "query":
            user_query = params.get("query")
            codebases = pipeline_main(user_query)
            response = {"result": codebases}
            await self.send_message(websocket, response, session_id)
            await self.send_message(websocket, ConnectionManager.END_OF_MESSAGE_RESPONSE, session_id)
            logger.info("generate_subtasks result: %s", codebases)

        elif method == "shutdown":
            self.agent_managers[session_id].shutdown()
            await self.send_message(websocket, {"result": "shutdown"}, session_id)
            await self.send_message(websocket, ConnectionManager.END_OF_MESSAGE_RESPONSE, session_id)
            self.agent_managers.pop(session_id)
            logger.info("Shutdown initiated")
            #os.kill(os.getpid(), signal.SIGTERM)

    async def send_message(self, websocket: WebSocket,
                           message: Dict[str, Any], session_id: str) -> None:
        """
        Sends a message to a specific WebSocket connection. If the
        connection is disconnected, the message is buffered.

        :param websocket: The WebSocket connection to send the message
            to.
        :param message: The message to send.
        """
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            self.message_buffer[session_id].append(message)
            logger.warning("Message buffered due to disconnection")

    async def send_buffered_messages(
            self,
            websocket: WebSocket,
            session_id: str) -> None:
        """
        Sends all buffered messages to a specific WebSocket connection.

        :param websocket: The WebSocket connection to send the buffered
            messages to.
        """
        if session_id not in self.message_buffer:
            return

        while self.message_buffer[session_id]:
            buffered_message = self.message_buffer[session_id].pop(0)
            await self.send_message(websocket, buffered_message, session_id)
            logger.info("Sent buffered message: %s", buffered_message)

    async def send_keepalive_pings(
            self,
            websocket: WebSocket,
            session_id: str) -> None:
        """
        Sends keepalive pings to a specific WebSocket connection at
        regular intervals.

        :param websocket: The WebSocket connection to send the
            keepalive pings to.
        """
        while True:
            await asyncio.sleep(10)  # Adjust the interval as needed
            await self.send_message(websocket, self.KEEP_ALIVE_PING, session_id)
            logger.debug("Sent keepalive ping")

    async def websocket_endpoint(
            self,
            websocket: WebSocket,
            session_id: str) -> None:
        """
        WebSocket endpoint to handle various agent management tasks.

        This endpoint manages WebSocket connections and processes
        incoming messages to perform tasks such as initializing
        external repo agents, generating and fine-tuning subtasks,
        running subtasks, and shutting down the agent manager.

        :param websocket: The WebSocket connection instance.
        :param session_id: The session identifier for the connection.

        :raises WebSocketDisconnect: If the WebSocket connection is
            disconnected.
        """
        await self._on_connect(websocket, session_id)
        keepalive_task = asyncio.create_task(
            self.send_keepalive_pings(websocket, session_id))

        try:
            await self.send_buffered_messages(websocket, session_id)

            while True:
                data = await websocket.receive_json()
                logger.info("Received data: %s", data)
                await self._on_receive(websocket, session_id, data)

        except WebSocketDisconnect:
            await self._on_disconnect(websocket)
        finally:
            keepalive_task.cancel()
            logger.info("Keepalive task cancelled")

    def ping(self):
        return 'pong'

def main():
    conn_manager = ConnectionManager()

    app = FastAPI()
    app.add_api_websocket_route(
        "/ws/{session_id}", conn_manager.websocket_endpoint)
    app.add_api_route("/ping", conn_manager.ping)

    parser = argparse.ArgumentParser(
        description="Run AgentManager with FastAPI WebSocket")
    parser.add_argument("--port", type=int, default=11111,
                        help="Port for the FastAPI server")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()