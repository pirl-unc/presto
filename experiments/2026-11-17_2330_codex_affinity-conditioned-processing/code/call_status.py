"""Best-effort Modal call graph; corroborate failures before any recovery."""
import argparse
import json

import modal
from common import E, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--call', required=True)
    args = parser.parse_args()
    def compact(node):
        return {'call': node.function_call_id, 'input': node.input_id, 'task': node.task_id,
                'function': node.function_name, 'status': node.status.name,
                'children': [compact(child) for child in node.children]}
    graph = [compact(node) for node in modal.FunctionCall.from_id(args.call).get_call_graph()]
    dump(E / 'results' / f'call-status-{args.call}.json', graph)
    print(json.dumps(graph))


if __name__ == '__main__':
    main()
