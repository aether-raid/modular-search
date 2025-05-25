# tests/test_web_raider.py

import asyncio
from websockets.sync.client import connect
import json

def test():
    with connect('ws://localhost:11111/ws/dumbass') as websocket:
        query = 'How do I parse Javascript AST in Python with Tree-Sitter?'
        message = ({
            'method': 'query',
            'params': {'query': query}
        })
        
        websocket.send(json.dumps(message))
        response = websocket.recv()
        asyncio.log(response)

if __name__ == '__main__':
    asyncio.run(test())