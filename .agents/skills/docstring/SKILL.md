# Skill: Python Docstring Generator

## Description
This skill analyzes Python source code (functions, classes, and methods) to generate detailed, standardized docstrings following the **Google Python Style Guide**.

## Usage Context
Apply this skill whenever a user requests code documentation, or when a Python file/block is identified as lacking documentation. It ensures consistency across research modules and production scripts.

## Style Guidelines
* **Format:** Google Style Docstrings.
* **Language:** English.
* **Type Hinting:** Extract types directly from Python type hints. If hints are missing, infer types from the logic or mark them as `Any`.
* **Indentation:** Strictly maintain the 4-space indentation standard.

## Docstring Components
1.  **Summary:** A concise, one-line description of the purpose (imperative mood).
2.  **Extended Description:** (Optional) Deeper explanation of the logic or mathematical background.
3.  **Args:** List of arguments with name, type (in parentheses), and description.
4.  **Returns:** Description of the return value and its type.
5.  **Raises:** Explicitly list any errors or exceptions raised by the code.
6.  **Example:** A `>>>` code block for complex logic or API usage.

## Few-Shot Examples

**Input:**
```python
def train_hmc_model(data_loader, model, optimizer, epochs=10):
    # training logic
    return history