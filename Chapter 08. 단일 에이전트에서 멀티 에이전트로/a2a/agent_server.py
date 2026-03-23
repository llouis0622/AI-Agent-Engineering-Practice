import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

agent_card = {
    "name": "SummarizerAgent",
    "description": "텍스트 요약을 수행하는 AI 에이전트입니다.",
    "protocolVersion": "1.0",
    "url": "http://localhost:8000",
    "provider": {
        "organization": "Example Org",
        "url": "https://example.org"
    },
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False
    },
    "skills": [
        {
            "id": "summarize-text",
            "name": "텍스트 요약",
            "description": "주어진 텍스트를 간결하게 요약합니다.",
            "tags": ["summarization", "nlp", "text-processing"],
            "examples": [
                "이 기사를 요약해 주세요",
                "다음 내용을 간단히 정리해 주세요"
            ]
        }
    ],
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "security": []
}


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/.well-known/agent-card.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(agent_card, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            rpc_request = json.loads(post_data)

            if rpc_request.get('jsonrpc') == '2.0' and rpc_request['method'] == 'message/send':
                params = rpc_request.get('params', {})
                message = params.get('message', {})
                parts = message.get('parts', [])

                text = ""
                for part in parts:
                    if 'text' in part:
                        text += part['text']

                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                try:
                    llm_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "당신은 간결한 요약을 제공하는 유용한 어시스턴트입니다."},
                            {"role": "user", "content": f"다음 텍스트를 요약하세요:\n{text}"}
                        ],
                    )
                    summary = llm_response.choices[0].message.content.strip()
                except Exception as e:
                    summary = f"요약 중 오류 발생: {str(e)}"

                task_id = str(uuid.uuid4())
                response = {
                    "jsonrpc": "2.0",
                    "result": {
                        "id": task_id,
                        "contextId": params.get('contextId', str(uuid.uuid4())),
                        "status": {
                            "state": "completed"
                        },
                        "artifacts": [
                            {
                                "parts": [{"text": summary}]
                            }
                        ]
                    },
                    "id": rpc_request['id']
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            else:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": rpc_request.get('id')
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, AgentHandler)
    print("A2A 에이전트 서버를 시작합니다. 주소: http://localhost:8000")
    print("Agent Card: http://localhost:8000/.well-known/agent-card.json")
    httpd.serve_forever()
