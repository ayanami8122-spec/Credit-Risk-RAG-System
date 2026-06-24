import numexpr as ne
from langchain_core.tools import Tool

def evaluate_math_expression(expression: str) -> str:
    """
    安全地计算数学表达式。
    """
    try:
        # 清理可能传入特殊字符
        expression = expression.replace('=', '').strip()
        # 使用 numexpr 进行计算
        result = ne.evaluate(expression)
        # 转换为字符串返回
        return str(result.item() if hasattr(result, "item") else result)
    except Exception as e:
        return f"计算失败，请确保输入的是纯数学表达式（如 '(100)/(50+20+10)'）。错误信息: {str(e)}"

# 封装为 Agent 可用的 Tool
financial_calculator_tool = Tool(
    name="Financial_Calculator",
    func=evaluate_math_expression,
    description="""一个精确的数学计算器。
    当你需要计算具体的金融公式（如资本充足率 CAR、Z-Score、违约距离 DD、杠杆率等）时，必须调用此工具。
    【注意】：传入的参数必须是带入具体数字后的合法纯数学表达式字符串！
    例如：'(100) / (50 + 20 + 10)' 或 '1.2*0.5 + 1.4*0.2'，不要传入中文字符或变量名。"""
)