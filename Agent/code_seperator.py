import os
import ast
import click
import astor
from collections import defaultdict

class CodeSplitter(ast.NodeVisitor):
    """Splits a Python file into separate files based on function and class method definitions."""

    def __init__(self, original_filename):
        self.original_filename = original_filename
        self.output_files = []
        self.current_class = None
        self.imports = []
        self.from_imports = []
        self.file_counter = 0
        self.global_code = []
        self.processed_classes = set()
        self.imported_names = defaultdict(list)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((alias.name, f"import {alias.name}\n"))

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.from_imports.append((alias.name, f"from {node.module} import {alias.name}\n"))

    def visit_ClassDef(self, node):
        self.current_class = node
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class:
            self.extract_method(node)
        else:
            self.extract_function(node)

    def visit(self, node):
        if isinstance(node, ast.Module):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                    self.visit(child)
                else:
                    self.global_code.append(child)
        else:
            super().visit(node)

    def extract_function(self, node):
        used_imports = self.get_used_imports(node)
        new_tree = ast.Module(body=used_imports + [node])
        new_content = astor.to_source(new_tree)
        new_filename = f"./{self.original_filename.split('.')[0]}_{self.file_counter}_v0.py"
        self.output_files.append((new_filename, new_content))
        self.file_counter += 1

    def extract_method(self, node):
        class_body = [node]

        # Include docstring only if this is the first method of the class
        if self.current_class.name not in self.processed_classes:
            class_docstring = ast.get_docstring(self.current_class)
            if class_docstring:
                docstring_node = ast.Expr(value=ast.Constant(value=class_docstring))
                class_body.insert(0, docstring_node)
            self.processed_classes.add(self.current_class.name)

        class_copy = ast.ClassDef(
            name=self.current_class.name,
            bases=self.current_class.bases,
            keywords=self.current_class.keywords,
            body=class_body,
            decorator_list=[]
        )
        used_imports = self.get_used_imports(class_copy)
        new_tree = ast.Module(body=used_imports + [class_copy])
        new_content = astor.to_source(new_tree)
        new_filename = f"./{self.original_filename.split('.')[0]}_{self.file_counter}_v0.py"
        self.output_files.append((new_filename, new_content))
        self.file_counter += 1

    def extract_global_code(self):
        if self.global_code:
            global_module = ast.Module(body=self.global_code)
            used_imports = self.get_used_imports(global_module)
            new_tree = ast.Module(body=used_imports + self.global_code)
            new_content = astor.to_source(new_tree)
            new_filename = f"./{self.original_filename.split('.')[0]}_global_v0.py"
            self.output_files.append((new_filename, new_content))

    def get_used_imports(self, node):
        used_names = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                used_names.add(n.id)
            elif isinstance(n, ast.Attribute):
                used_names.add(n.attr)
            elif isinstance(n, ast.ClassDef):
                for base in n.bases:
                    if isinstance(base, ast.Name):
                        used_names.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        used_names.add(base.attr)

        used_imports = []
        for name, import_stmt in self.imports + self.from_imports:
            if name in used_names:
                used_imports.append(import_stmt)

        return used_imports

def split_python_file(dir_path, output_dir):
    """
    Splits a Python file into separate files based on function and class method definitions.

    Args:
        dir_path (str): The path to the directory containing the Python files to be split.
        output_dir (str): The directory where the split files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as file:
                    content = file.read()
                tree = ast.parse(content)
                splitter = CodeSplitter(os.path.basename(file_path))
                splitter.visit(tree)
                splitter.extract_global_code()  # Extract global code after visiting all nodes
                for filename, content in splitter.output_files:
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w') as f:
                        f.write(content)
                    print(f"Created file: {output_path}")

@click.command()
@click.argument('dir_path', type=str, default='.', required=True)
@click.option('--output-dir', '-o', type=click.Path(), default='.',
              help='Directory to output the split files (default: current directory)')
def main(dir_path, output_dir):
    """
    Splits a Python file into separate files based on function and class method definitions.

    Args:
        dir_path (str): The path to the directory containing the Python files to be split.
        output_dir (str): The directory where the split files will be saved.
    """
    split_python_file(dir_path, output_dir)

if __name__ == "__main__":
    main()