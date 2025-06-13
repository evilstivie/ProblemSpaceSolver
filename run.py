import json
import itertools
import sys
import typing

import asyncclick as click
import fastmcp

from problem_space.tasks import game24
from problem_space.methods import iterative, cot


@click.group()
async def cli():
    pass


@cli.command()
@click.option('--model', type=str, default='cogito:14b')
@click.option('--temperature', type=float, default=0.1)
@click.option('--task-idx-from', type=int, default=0)
@click.option('--num-tasks', type=int, default=20)
@click.option('--output', type=click.File(mode="a"), default="output.json")
async def run_experiment(
    model: str,
    temperature: float,
    task_idx_from: int,
    num_tasks: int,
    output: typing.IO,
):
    output.write(model+"_"+str(temperature)+"\n")
    for i, task in itertools.islice(enumerate(game24.iter_tasks()), task_idx_from, task_idx_from + num_tasks):
        for p in range(3):
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
                answer, messages = await iterative.run(
                    client,
                    task,
                    model=model,
                    temperature=temperature,
                    max_iter=50,
                )

            is_solved = task.validate(answer)
            print("IS_SOLVED:", is_solved)
            output.write(json.dumps({
                'i': i,
                'p': p,
                'method': 'problem_space',
                'task': task.input,
                'answer': answer,
                'is_solved': int(is_solved),
                'chat': messages,
            }) + "\n")

            answer, messages = await cot.run(
                task,
                model=model,
                temperature=0.5,
            )

            is_solved = task.validate(answer)
            print("IS_SOLVED:", is_solved)
            output.write(json.dumps({
                'i': i,
                'p': p,
                'method': 'cot',
                'task': task.input,
                'answer': answer,
                'is_solved': int(is_solved),
                'chat': messages,
            }) + "\n")


@cli.command()
async def run_model_mcp():
    from problem_space.problem_space.mcp import mcp
    await mcp.run_async()


if __name__ == '__main__':
    cli()
