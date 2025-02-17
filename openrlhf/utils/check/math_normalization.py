# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
import pdb
import re
import sympy
import threading
from sympy.parsing.latex import parse_latex
from .qwen_equal import math_equal
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}'), ('Thefinalansweris', ''),
    ("bits", ""), ('/\u03c0Hz', ""), ('\\times10^5N', ""), ("Hz", ""), ("dfrac", "frac"), ("tfrac", "frac")
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots',
]

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = str(final_answer).split('=')[-1]
    if final_answer.endswith("\\"): final_answer=final_answer[:-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
    # 3.0 -> 3
    if final_answer.endswith(".0") and final_answer[:-2].isdigit():
        final_answer = final_answer[:-2]
    # 3.00 -> 3
    if final_answer.endswith(".00") and final_answer[:-3].isdigit():
        final_answer = final_answer[:-3]
    if final_answer.endswith("%") and final_answer[:-1].isdigit():
        final_answer = final_answer[:-1]
    # A -> a
    if final_answer.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
        final_answer = final_answer.lower()
    return final_answer

def compare_rounded_numbers(str1, str2):
    # 尝试将字符串转换为浮点数
    if "." not in str1 or "." not in str2:
        return False  # 如果两个字符串都没有小数点，直接返回False
    try:
        num1 = float(str1)
        num2 = float(str2)
    except ValueError:
        return False  # 如果转换失败，返回False
    if num2 !=0:
        if abs(num1 - num2 / (num2))<0.0001: return True
    # 获取第二个数字的小数位数
    decimal_places = len(str2.split('.')[1]) if '.' in str2 else 0

    # 将第一个数字四舍五入到第二个数字的小数位数
    # rounded_num1 = round(num1, decimal_places)

    # 将第二个数字四舍五入到相同的小数位数
    # rounded_num2 = round(num2, decimal_places)

    # 比较两个四舍五入后的数字
    return num1 == num2


def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    # 四舍五入
    # special case:
    if compare_rounded_numbers(formatted_target_str, formatted_prediction_str): return True
    if compare_rounded_numbers(formatted_prediction_str, formatted_target_str): return True

    if formatted_target_str=="True": 
        if "yes" in formatted_prediction_str.lower(): return True
    if formatted_target_str=="False" : 
        if "no" in formatted_prediction_str.lower(): return True
    if len(formatted_prediction_str) >= 40:
        return False
    flag = False    
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True
    
    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True
    
    if flag == True:
        return formatted_target_str == formatted_prediction_str


    if math_equal(formatted_target_str, formatted_prediction_str): return True
    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False

if __name__ == "__main__":
    pred="0.93Hz"
    gt="0.9"
    pred=normalize_final_answer(pred)
    gt=normalize_final_answer(gt)
    check_sympy_equivalence(pred, gt)