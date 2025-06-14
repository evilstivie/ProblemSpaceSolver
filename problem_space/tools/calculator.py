from typing import Annotated

import fastmcp
from pydantic import Field
import sympy

mcp = fastmcp.FastMCP(name="Calculator")


@mcp.tool()
def evaluate_expression(
    expression: Annotated[str, Field(description="Expression you want to evaluate")],
) -> float:
    """
    Use this tool for precise math calculations. Useful for calculation correctness verification.

    EXAMPLES:
    - expression="1 + 2 * 3"
      answer=7

    - expression="1 + (3/2 * 10)"
      answer=16
    """
    try:
        return sympy.simplify(sympy.parse_expr(expression))
    except sympy.SympifyError as err:
        raise Exception(str(err) + ": " + err.expr)
