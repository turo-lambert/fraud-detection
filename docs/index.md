# Skaff repo template

This template provides the basic structure and components for accelerator developement on Skaff.

In the box:

- An `mkdocs.yaml` file and `docs` directory to document your accelerator.
- A GitHub action to deploy this doc on GH pages.
- A `.gitignore`
- An apache 2.0 licence.

The goal is for it to be as light as possible. It does not include some elements that one might consider good practice such as hooks, tests, or other highly opinionated folder structure.
It does not mean these are not expected, just not from day 1 and not for all projects.


## Documentation features and tricks

### [Admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/)
=== "Result"

    !!! note "This is a note"
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
        nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
        massa, nec semper lorem quam in massa.

=== "Markdown"

    ```
    !!! note "this is a note"
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
        nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
        massa, nec semper lorem quam in massa.
    ```

### [Code annotations](https://squidfunk.github.io/mkdocs-material/reference/annotations/)

=== "Result"

    ```python
    def foo(bar):
        return "foo" + bar #(1)!
    ```

    1. You can explain a line of code here.

=== "Markdown"

    ``````

    ```python
    def foo(bar):
        return "foo" + bar #(1)!
    ```

    1. You can explain a line of code here.
    ``````

### [Code highlighting](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/?h=code+hi#highlighting-specific-lines-line-ranges)

=== "Result"
    ``` py hl_lines="2 3"
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```

=== "Markdown"
    ``````
    ``` py hl_lines="2 3"
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```
    ``````

### [Nice terminal](https://termynal.github.io/termynal.py/)

=== "Result"

    <!-- termynal -->
    ```
    $ python script.py
    ```

=== "Markdown"

    Remove the `#` for it to work properly
    ``````
    #<!-- termynal -->
    ```
    $ python script.py
    ```
    ``````
