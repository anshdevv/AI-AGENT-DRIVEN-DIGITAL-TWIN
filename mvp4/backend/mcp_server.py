from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import orchestrator


router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPRequest(BaseModel):
    method: str
    params: dict = Field(default_factory=dict)
    id: str | int | None = None


@router.get("")
def mcp_info() -> dict:
    return {
        "name": "customer-service-mcp",
        "protocol": "lightweight-jsonrpc",
        "methods": ["tools/list", "tools/call"],
    }


@router.post("")
def handle_mcp(request: MCPRequest) -> dict:
    if request.method == "tools/list":
        return {"id": request.id, "result": {"tools": orchestrator.tools.tool_definitions()}}

    if request.method == "tools/call":
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})
        result = orchestrator.tools.call_tool(tool_name, arguments)
        if not result.ok:
            raise HTTPException(status_code=400, detail=result.error)
        return {"id": request.id, "result": result.data}

    raise HTTPException(status_code=400, detail=f"Unsupported MCP method: {request.method}")
