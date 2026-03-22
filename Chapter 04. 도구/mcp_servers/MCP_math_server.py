import sys
import json
import ast
import operator
from typing import Any, Dict

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = eval_expr(node.left)
        right = eval_expr(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        operand = eval_expr(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def compute_math(expression: str) -> float:
    """안전하게 산술 표현식을 파싱하고 평가합니다."""
    try:
        cleaned = "".join(ch for ch in expression if ch.isdigit() or ch in "+-*/()^ .*")
        cleaned = cleaned.replace("^", "**").strip()
        if not cleaned:
            raise ValueError("수식을 찾을 수 없습니다.")

        expr_ast = ast.parse(cleaned, mode="eval").body
        return eval_expr(expr_ast)
    except Exception as e:
        raise ValueError(f"수식 '{expression}' 계산 오류: {e}")


def handle_jsonrpc_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-RPC 2.0 요청을 처리합니다."""
    jsonrpc = request.get("jsonrpc")
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params", {})

    if jsonrpc != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32600,
                "message": "Invalid Request: jsonrpc must be '2.0'"
            }
        }

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "math-server",
                        "version": "1.0.0"
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "math",
                            "description": "수학 표현식을 계산합니다. 예: (3 + 5) * 12",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "expression": {
                                        "type": "string",
                                        "description": "계산할 수학 표현식"
                                    }
                                },
                                "required": ["expression"]
                            }
                        }
                    ]
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "math":
                expression = arguments.get("expression", "")
                result = compute_math(expression)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"계산 결과: {result}"
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


def main():
    """stdin에서 JSON-RPC 요청을 읽고 stdout으로 응답을 출력합니다."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)

            if "id" not in request or request.get("id") is None:
                continue

            response = handle_jsonrpc_request(request)
        except json.JSONDecodeError as e:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}"
                }
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
