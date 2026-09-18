import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import ToolNode

from ai.graphs import tool_agent_graph as graph


def call(name='get_transaction', identifier='call-1', **args):
    return {'name': name, 'args': args, 'id': identifier}


@pytest.fixture
def fake_tools(monkeypatch):
    calls = []

    @tool
    def get_transaction(transaction_id: str) -> dict:
        """Read a transaction."""
        calls.append(transaction_id)
        return {'found': False, 'transaction': None}

    monkeypatch.setattr(graph, 'tool_node', ToolNode([get_transaction]))
    return calls


def test_general_avoids_tools_and_logs_no_content(monkeypatch, caplog):
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content='A streaming platform.')
    monkeypatch.setattr(graph, 'model_with_tools', model)
    tools = AsyncMock()
    monkeypatch.setattr(graph, 'tool_node', tools)
    with caplog.at_level(logging.INFO, logger=graph.__name__):
        result = asyncio.run(graph.run_agent_graph('Private question', 'REQ-general'))
    assert [type(m) for m in result['messages']] == [HumanMessage, AIMessage]
    assert result['tools_used'] == []
    assert isinstance(model.ainvoke.call_args.args[0][0], SystemMessage)
    tools.ainvoke.assert_not_awaited()
    assert 'node=agent' in caplog.text
    assert 'graph=end' in caplog.text
    assert 'Private question' not in caplog.text


@pytest.mark.parametrize('parallel', [True, False])
def test_multiple_calls_preserve_history_and_ids(monkeypatch, fake_tools, parallel):
    first = call(transaction_id='TX-100')
    second = call(identifier='call-2', transaction_id='TX-200')
    responses = [AIMessage(content='', tool_calls=[first, second])] if parallel else [
        AIMessage(content='', tool_calls=[first]), AIMessage(content='', tool_calls=[second]),
    ]
    responses.append(AIMessage(content='Neither transaction was found.'))
    model = AsyncMock()
    model.ainvoke.side_effect = responses
    monkeypatch.setattr(graph, 'model_with_tools', model)
    result = asyncio.run(graph.run_agent_graph('Compare TX-100 and TX-200', 'REQ-multi'))
    assert sorted(fake_tools) == ['TX-100', 'TX-200']
    assert result['tools_used'] == ['get_transaction', 'get_transaction']
    messages = result['messages']
    assert [m.tool_call_id for m in messages if isinstance(m, ToolMessage)] == ['call-1', 'call-2']
    assert messages[-1].content == 'Neither transaction was found.'
    assert len(model.ainvoke.call_args.args[0]) == len(messages)


def test_unknown_tool_never_executes(monkeypatch, fake_tools):
    model = AsyncMock()
    model.ainvoke.side_effect = [
        AIMessage(content='', tool_calls=[call('delete_database')]),
        AIMessage(content='That tool is unavailable.'),
    ]
    monkeypatch.setattr(graph, 'model_with_tools', model)
    result = asyncio.run(graph.run_agent_graph('Delete everything', 'REQ-unknown'))
    assert fake_tools == []
    tool_message = result['messages'][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == 'error'
    assert 'not a valid tool' in tool_message.content


def test_loop_is_bounded(monkeypatch, fake_tools):
    model = AsyncMock()
    model.ainvoke.side_effect = lambda *args, **kwargs: AIMessage(
        content='', tool_calls=[call(transaction_id='TX-100')],
    )
    monkeypatch.setattr(graph, 'model_with_tools', model)
    with pytest.raises(GraphRecursionError):
        asyncio.run(graph.run_agent_graph('Repeat forever', 'REQ-limit'))
    assert model.ainvoke.await_count <= 4
    assert len(fake_tools) <= 4


def test_only_approved_read_tools_registered():
    assert set(graph.tool_node.tools_by_name) == {
        'get_transaction', 'get_top_products', 'get_revenue_summary', 'get_transaction_metrics',
    }


def test_tool_failures_do_not_expose_exception(monkeypatch):
    from ai.tools import transaction_tools

    def fail(_):
        raise RuntimeError('private database credentials')

    monkeypatch.setattr(transaction_tools, 'find_transaction', fail)
    model = AsyncMock()
    model.ainvoke.side_effect = [
        AIMessage(content='', tool_calls=[call(transaction_id='TX-123')]),
        AIMessage(content='Service unavailable.'),
    ]
    monkeypatch.setattr(graph, 'model_with_tools', model)
    result = asyncio.run(graph.run_agent_graph('Status of TX-123', 'REQ-fail'))
    assert result['messages'][2].status == 'error'
    assert 'private database credentials' not in result['messages'][2].content
