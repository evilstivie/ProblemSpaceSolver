import itertools
import sys

import asyncclick as click
import fastmcp

from problem_space.tasks import game24
from problem_space.methods import iterative


@click.group()
async def cli():
    pass


@cli.command()
async def run_experiment():
    stats = []
    for task in itertools.islice(game24.iter_tasks(), 10):
        config = {
            "mcpServers": {
                "cognitive_map": {
                    "command": sys.executable,
                    "args": [__file__, "run-model-mcp"],
                    "env": {},
                },
            }
        }
        client = fastmcp.Client(config)

        async with client:
            answer = await iterative.run(client, task)

        is_solved = task.validate(answer)
        stats.append((task.input, answer, is_solved))

    click.echo(stats)


@cli.command()
async def run_model_mcp():
    from problem_space.problem_space.mcp import mcp
    await mcp.run_async()


if __name__ == '__main__':
    cli()
