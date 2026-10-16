"""Benchmark tasks. Each has a prompt, optional starter files, and hidden pytest tests
that are copied into the workspace only after the agent has finished."""

from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    prompt: str
    hidden_test: str
    starter: dict[str, str] = field(default_factory=dict)
    kind: str = "write"   # "write" (new code) or "fix" (debug existing code)


TASKS = [
    Task("fizzbuzz", """Create fizzbuzz.py with a function fizzbuzz(n) that returns a list of strings for 1..n:
"Fizz" for multiples of 3, "Buzz" for multiples of 5, "FizzBuzz" for both, otherwise the number as a string.
Run it to check.""", """
from fizzbuzz import fizzbuzz
def test_fizzbuzz():
    assert fizzbuzz(15) == ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
    assert fizzbuzz(0) == []
"""),

    Task("primes", """Create primes.py with is_prime(n) (False for n < 2) and primes_up_to(n) returning a sorted list of
all primes <= n. Verify it by running the file.""", """
from primes import is_prime, primes_up_to
def test_primes():
    assert [n for n in range(-3, 20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
    assert primes_up_to(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert primes_up_to(1) == []
"""),

    Task("fix_average", """average.py crashes when run. Run it, read the error, find the cause and fix the file.
average([]) should return 0. Run it again to confirm.""", """
from average import average
def test_average():
    assert average([10, 20, 30, 40]) == 25
    assert average([5]) == 5
    assert average([]) == 0
""", starter={"average.py": """def average(numbers):
    total = 0
    for i in range(len(numbers) + 1):
        total += numbers[i]
    return total / len(numbers)


if __name__ == "__main__":
    print("Average:", average([10, 20, 30, 40]))
"""}, kind="fix"),

    Task("fix_discount", """discount.py has a logic bug: apply_discount(price, percent) should return the price reduced by
percent percent, rounded to 2 decimals (apply_discount(200, 15) == 170.0). percent outside 0-100 must raise ValueError.
Fix it and verify.""", """
import pytest
from discount import apply_discount
def test_discount():
    assert apply_discount(200, 15) == 170.0
    assert apply_discount(99.99, 0) == 99.99
    assert apply_discount(50, 100) == 0.0
    with pytest.raises(ValueError):
        apply_discount(10, 120)
    with pytest.raises(ValueError):
        apply_discount(10, -5)
""", starter={"discount.py": """def apply_discount(price, percent):
    return round(price - percent, 2)


if __name__ == "__main__":
    print(apply_discount(200, 15))
"""}, kind="fix"),

    Task("calculator", """Build calculator.py with add, subtract, multiply and divide. divide must raise ValueError on
division by zero. Also write test_calculator.py with pytest tests for all four (including divide by zero) and run them.""",
         """
import pytest
import calculator as c
def test_calc():
    assert c.add(2, 3) == 5 and c.subtract(2, 3) == -1 and c.multiply(4, 2.5) == 10 and c.divide(9, 3) == 3
    with pytest.raises(ValueError):
        c.divide(1, 0)
"""),

    Task("word_count", """Create text_stats.py with word_counts(text) returning a dict of lowercase word -> count.
Ignore punctuation (word_counts("Hi, hi! Bye.") == {"hi": 2, "bye": 1}) and keep apostrophes inside words like "don't".
Test it.""", """
from text_stats import word_counts
def test_counts():
    assert word_counts("Hi, hi! Bye.") == {"hi": 2, "bye": 1}
    assert word_counts("I don't know. Don't ask!") == {"i": 1, "don't": 2, "know": 1, "ask": 1}
    assert word_counts("") == {}
"""),

    Task("csv_summary", """sales.csv has columns region,amount. Write report.py with summarize(path) that returns a dict
mapping each region to its total amount (as float), using only the standard library. Run it on sales.csv.""", """
from report import summarize
def test_summary():
    assert summarize("sales.csv") == {"North": 350.0, "South": 120.5, "East": 80.0}
""", starter={"sales.csv": "region,amount\nNorth,100\nSouth,120.5\nNorth,250\nEast,80\n"}),

    Task("stack", """Create stack.py with a Stack class: push(x), pop() (raises IndexError when empty), peek() (raises
IndexError when empty), is_empty() and __len__. Write your own pytest tests and make sure they pass.""", """
import pytest
from stack import Stack
def test_stack():
    s = Stack()
    assert s.is_empty() and len(s) == 0
    s.push(1); s.push(2)
    assert s.peek() == 2 and len(s) == 2
    assert s.pop() == 2 and s.pop() == 1 and s.is_empty()
    with pytest.raises(IndexError):
        s.pop()
    with pytest.raises(IndexError):
        s.peek()
"""),
]
