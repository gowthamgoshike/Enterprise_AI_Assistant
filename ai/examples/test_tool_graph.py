"""Run a live request and print node/tool names without dumping tool data."""

import argparse
import asyncio
import uuid

from langchain_core.messages import HumanMessage

from ai.graphs.tool_agent_graph import enterprise_tool_graph, extract_tools_used


async def main(user_message: str):
    initial_state = {
        'messages': [HumanMessage(content=user_message)],
        'request_id': str(uuid.uuid4()),
        'route': None,
        'current_turn_tools': [],
        'error': None,
    }
    messages = []
    async for event in enterprise_tool_graph.astream(
        initial_state,
        config={"configurable": {"thread_id": "example-thread"}},
        stream_mode='updates',
    ):
        for node, update in event.items():
            print(f'node={node}')
            for message in update.get('messages', []):
                messages.append(message)
                for call in getattr(message, 'tool_calls', []) or []:
                    print(f'  requested_tool={call["name"]}')
                if message.type == 'tool':
                    print(f'  tool={message.name} status={message.status}')
    print('graph=end')
    print('tools_used=', extract_tools_used(messages))
    print(messages[-1].text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?', default='Explain Kafka in one sentence.')
    asyncio.run(main(parser.parse_args().prompt))
