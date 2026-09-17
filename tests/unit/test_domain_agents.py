import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from ai.agents import analytics_agent, general_agent, orchestrator, transaction_agent
from ai.schemas.agent import IntentRoute


@pytest.mark.parametrize('route', ['transaction', 'analytics', 'general'])
def test_orchestrator_dispatches_only_selected_domain(monkeypatch, route):
    monkeypatch.setattr(orchestrator, 'route_request', AsyncMock(return_value=IntentRoute(
        route=route, confidence=0.9, reason='Observable classification rationale.'
    )))
    handlers = {}
    for domain in ['transaction', 'analytics', 'general']:
        result = 'answer' if domain == 'general' else {'response': 'answer', 'tools_used': ['tool']}
        handlers[domain] = AsyncMock(return_value=result)
        monkeypatch.setattr(orchestrator, f'handle_{domain}_request', handlers[domain])
    result = asyncio.run(orchestrator.process_request('question'))
    assert result['route'] == route
    assert result['route_confidence'] == 0.9
    assert result['response'] == 'answer'
    for domain, handler in handlers.items():
        assert handler.await_count == (1 if domain == route else 0)
    if route == 'general':
        assert result['tools_used'] == []


@pytest.mark.parametrize('module,domain,allowed,forbidden', [
    (transaction_agent, 'transaction', 'get_transaction', 'get_top_products'),
    (analytics_agent, 'analytics', 'get_top_products', 'get_transaction'),
])
def test_domain_tool_execution_and_restrictions(monkeypatch, module, domain, allowed, forbidden):
    model = AsyncMock()
    tool = AsyncMock()
    tool.ainvoke.return_value = {'found': True}
    monkeypatch.setattr(module, f'{domain}_model', model)
    monkeypatch.setattr(module, f'{domain.upper()}_TOOL_MAP', {allowed: tool})
    handler = getattr(module, f'handle_{domain}_request')
    model.ainvoke.side_effect = [
        AIMessage(content='', tool_calls=[{'name': allowed, 'args': {}, 'id': 'call-1'}]),
        AIMessage(content='answer'),
    ]
    result = asyncio.run(handler('question'))
    assert result == {'response': 'answer', 'tools_used': [allowed]}
    messages = model.ainvoke.call_args.args[0]
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == 'call-1'
    tool.ainvoke.assert_awaited_once_with({})

    tool.reset_mock()
    model.ainvoke.side_effect = [AIMessage(
        content='', tool_calls=[{'name': forbidden, 'args': {}, 'id': 'call-2'}]
    )]
    with pytest.raises(ValueError, match='not permitted'):
        asyncio.run(handler('question'))
    tool.ainvoke.assert_not_awaited()

    model.ainvoke.side_effect = [AIMessage(content='Please provide more detail.')]
    assert asyncio.run(handler('question'))['tools_used'] == []


def test_general_handler(monkeypatch):
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content='Kafka is an event streaming platform.')
    monkeypatch.setattr(general_agent, 'model', model)
    assert asyncio.run(general_agent.handle_general_request('Explain Kafka.')).startswith('Kafka')
    assert model.ainvoke.call_args.args[0][1].content == 'Explain Kafka.'


@pytest.mark.parametrize('confidence,dispatched', [(0.42, False), (0.6499, False), (0.65, True)])
def test_confidence_fallback_and_private_logs(monkeypatch, caplog, confidence, dispatched):
    import logging

    monkeypatch.setattr(orchestrator, 'route_request', AsyncMock(return_value=IntentRoute(
        route='analytics', confidence=confidence, reason='Private classification reason'
    )))
    handlers = []
    for domain in ['transaction', 'analytics', 'general']:
        handler = AsyncMock(return_value={'response': 'answer', 'tools_used': []})
        monkeypatch.setattr(orchestrator, f'handle_{domain}_request', handler)
        handlers.append(handler)
    with caplog.at_level(logging.INFO, logger=orchestrator.__name__):
        result = asyncio.run(orchestrator.process_request('Sensitive user message', request_id='test-request'))
    assert handlers[1].await_count == int(dispatched)
    handlers[0].assert_not_awaited()
    handlers[2].assert_not_awaited()
    assert result['route_confidence'] == confidence
    assert result['route'] == 'analytics'
    if not dispatched:
        assert 'clarify' in result['response']
        assert result['tools_used'] == []
    assert 'test-request' in caplog.text
    assert 'route=analytics' in caplog.text
    assert 'Sensitive user message' not in caplog.text
    assert 'Private classification reason' not in caplog.text
