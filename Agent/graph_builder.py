"""
This Python script analyzes and visualizes the structure of a Python codebase using the 
Abstract Syntax Tree (AST) module. It identifies Python files in a directory, parses them 
to construct their AST, and extracts information about the code's structure.

Key Features:
- Directory Traversal: Processes all Python (.py) files in a given directory.
- AST Parsing: Constructs ASTs for detailed analysis of the code structure.
- Structure Analysis: Extracts details about classes, functions, variables, imports, 
  inheritance, and function calls.
- Graph Generation: Creates visual representations of the codebase structure using Graphviz.
- Logging: Logs AST details in an indented format for easier understanding and debugging.
- JSON Output: Outputs the code structure into a JSON file for documentation or analysis.

Benefits:
- Provides insights into the architecture of Python projects.
- Assists in code comprehension and refactoring efforts.

Note:
This script is a utility tool for code analysis and should be part of a larger toolkit for 
codebase management and documentation.
"""

import re
import os
import copy
import ast
import json

import click

from typing import Union, List, Tuple
from collections import defaultdict
from pathlib import Path

from graphviz import Digraph

from prompt import ENTITY_EXTRACTION,RELATIONSHIP_EXTRACTION


class ASTPrinter(ast.NodeVisitor):
    """ASTPrinter is a class for printing and logging the details of an Abstract Syntax Tree (AST) 
    node and its children in a structured and indented format. It's useful for debugging and 
    understanding the structure of an AST.

    Attributes:
        indentation_level (int): Manages the current level of indentation for printing.
        log_file (file): The file where the AST details are logged.
    """

    def __init__(self, log_file_path: str):
        self.indentation_level = 0
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
            print(f"File: {log_file_path} removed successfully.")
        else:
            print(f"File: {log_file_path} does not exist.")
        # pylint: disable=W1514, R1732
        self.log_file = open(log_file_path, 'a')

    def __del__(self):
        """Destructor for ASTPrinter. Closes the log file."""
        if self.log_file:
            self.log_file.close()

    def log(self, message: str) ->None:
        """Logs a message to both the console and the log file.

        Args:
            message (str): The message to be logged.
        """
        self.log_file.write(message + '\n')

    def visit(self, node: Union[ast.AST, List[ast.AST]]) ->None:
        """Visits each node in the AST. If the node is a list, it recursively visits each
        item in the list. For AST nodes, it prints the node type and relevant details
        such as 'name' or 'id' if present. It also manages indentation for readability.

        Args:
            node (Union[ast.AST, List[ast.AST]]): The AST node or a list of AST nodes to visit.
        """
        if not isinstance(node, (ast.AST, list)):
            return
        if isinstance(node, list):
            for item in node:
                self.visit(item)
        else:
            indent = '  ' * self.indentation_level
            message = f"{indent}{type(node).__name__}"
            if hasattr(node, 'name'):
                message += f" (name: {node.name})"
            if hasattr(node, 'id'):
                message += f" (id: {node.id})"
            if hasattr(node, 'arg'):
                message += f" (arg: {node.arg})"
            if hasattr(node, 'attr'):
                message += f" (attr: {node.attr})"
            self.log(message)  # Log the message
            # Visit children of the current node
            self.indentation_level += 1
            self.generic_visit(node)
            self.indentation_level -= 1


class CodeVisitor(ast.NodeVisitor):
    """CodeVisitor is a class that traverses an AST (Abstract Syntax Tree) to extract
    information about the structure of the code. It collects details about classes,
    functions, variables, imports, and function calls, storing them in a graph structure.

    Attributes:
        filename (str): The name of the file being analyzed.
        graph (dict): A dictionary to store the extracted information from the code.
        current_scope (list): A stack to keep track of the current scope while traversing.
    """
    def __init__(self, filename: str, graph: dict) ->None:
        """Initializes the CodeVisitor with the filename of the code and an empty graph.

        Args:
            filename (str): The name of the file being analyzed.
            graph (dict): A dictionary to store the extracted information from the code.
        """
        self.filename = filename
        self.graph = graph
        self.current_scope = []  # Stack to keep track of the current scope
        self.call_depth = 0

    def build_key_from_current_scope(self) ->None:
        """
        Constructs a key string from the current scope stack. This key is used to identify 
        the current node's position in the code structure within the graph.

        The key is formed by concatenating the filename with the hierarchy of the current scope, 
        such as classes or functions, in which the node is nested.

        Returns:
            str: A string representing the unique key for the current scope.
        """
        key = "(file)" + self.filename + "->"
        for node in self.current_scope:
            key += node
            key += "->"
        return key[:-2]

    # pylint: disable=C0103
    def visit_ClassDef(self, node: ast.ClassDef) ->None:
        """Visits a ClassDef node in the AST and updates the graph with information about
        the class, its name, and its inheritance.

        Args:
            node (ast.ClassDef): The ClassDef node being visited.
        """
        # Update for handling inheritance
        base_classes = [base.id for base in node.bases if isinstance(base, ast.Name)]
        class_name = node.name
        self.current_scope.append("(class)" + class_name)
        self.graph[self.filename]['classes'].add(self.build_key_from_current_scope())
        for base in base_classes:
            self.graph[self.filename]['inheritance'][self.build_key_from_current_scope()].add(base)
        self.generic_visit(node)
        self.current_scope.pop()

    # pylint: disable=C0103
    def visit_FunctionDef(self, node: ast.FunctionDef) ->None:
        """Visits a FunctionDef node in the AST and updates the graph with information about
        the function.

        Args:
            node (ast.FunctionDef): The FunctionDef node being visited.
        """
        self.current_scope.append("(function)" + node.name)
        self.graph[self.filename]['functions'].add(self.build_key_from_current_scope())
        self.generic_visit(node)
        self.current_scope.pop()

    # pylint: disable=C0103
    def visit_Assign(self, node: ast.Assign) ->None:
        """Visits an Assign node in the AST and updates the graph with information about
        variable assignments.

        Args:
            node (ast.Assign): The Assign node being visited.
        """
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.graph[self.filename]['variables'].add(target.id)
        self.generic_visit(node)

    # pylint: disable=C0103
    def visit_Import(self, node: ast.Import) ->None:
        """Visits an Import node in the AST and updates the graph with information about
        import statements.

        Args:
            node (ast.Import): The Import node being visited.
        """
        for alias in node.names:
            self.graph[self.filename]['imports'].add(alias.name)
        self.generic_visit(node)

    # pylint: disable=C0103
    def visit_ImportFrom(self, node: ast.ImportFrom) ->None:
        """Visits an ImportFrom node in the AST and updates the graph with information about
        'from ... import ...' statements.

        Args:
            node (ast.ImportFrom): The ImportFrom node being visited.
        """
        module = node.module
        for alias in node.names:
            import_name = f"{module}.{alias.name}" if module else alias.name
            self.graph[self.filename]['imports'].add(import_name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """
        Visits a Name node in the AST. If the node is part of a function call (determined by the 
        call depth),it updates the graph with information about the function call.

        Args:
            node (ast.Name): The Name node being visited. This represents a variable or function 
            name in the code.
        """
        if self.call_depth != 0:
            name = node.id
            self.current_scope.append("(call)" + name)
            self.call_depth += 1
            self.generic_visit(node)
            self.graph[self.filename]['function_calls'].add(self.build_key_from_current_scope())
            self.call_depth = 0
            self.current_scope.pop()
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Visits an Attribute node in the AST. If the node is part of a function call (determined by 
        the call depth),it updates the graph with information about the attribute access, which 
        could be a method call.

        Args:
            node (ast.Attribute): The Attribute node being visited. This represents an attribute 
            reference, such as a method or property on an object.
        """
        if self.call_depth != 0:
            attribute = node.attr
            self.current_scope.append("(call)" + attribute)
            self.call_depth += 1
            self.generic_visit(node)
            self.graph[self.filename]['function_calls'].add(self.build_key_from_current_scope())
            self.call_depth = 0
            self.current_scope.pop()
        else:
            self.generic_visit(node)

    # pylint: disable=C0103
    def visit_Call(self, node: ast.Call) -> None:
        """Visits a Call node in the AST and updates the graph with information about
        function and method calls.

        Args:
            node (ast.Call): The Call node being visited.
        """
        self.call_depth += 1
        self.generic_visit(node)
        self.call_depth = 0

def check_entity_in_next_line(lines: List[str], entity_name: str, line_idx:int, line_num: int) ->Tuple[str, str]:
    """Checks if the given entity is in the next line of the provided lines list.

    Args:
        lines (List[str]): A list of strings where each string is a line of code.
        entity_name (str): The name of the entity to check.
        line_idx (int): The index of the current line in the lines list.
        line_num (int): The total number of lines in the lines list.

    Returns:
        Tuple[str, str]: A tuple where the first element is the entity name (prepended with "self." if the entity
        is in the next line and its name is "self") and the second element is the string "attribute".
    """     
    if line_idx + 1 < line_num:
        next_line = lines[line_idx + 1]
        if "(id:" in next_line:
            next_entity_name = next_line.strip().split("id:")[1].split(")")[0].strip()
            if next_entity_name == "self":
                return "self." + entity_name, "attribute"
    return entity_name, "attribute"

def replace_file_path(original_string: str, replacement: str) ->str:
    """Replaces the file path in the original string with the replacement string.

    This function uses regular expressions to find the file path in the original string,
    which is expected to be in the format '(file)<filepath>->'. It then replaces the 
    filepath with the replacement string.

    Args:
        original_string (str): The original string containing the file path.
        replacement (str): The string to replace the file path with.

    Returns:
        str: The modified string with the file path replaced.
    """     
    pattern = r'\(file\)(.*?)->'
    return re.sub(pattern, f'(file){replacement}->', original_string)

def extract_entities_and_relationships(log_file_path: str, source_file: str, graph: dict) ->None:
    """Extracts entities and their relationships from a source file and logs them.

    This function reads the log file generated from the AST of the source file, 
    extracts entities (like classes, functions, variables) and their relationships 
    (like inheritance, function calls), and writes them into an output file. 
    It also updates the graph dictionary with the extracted information.

    Args:
        log_file_path (str): The path to the log file generated from the AST of the source file.
        source_file (str): The path to the source file being analyzed.
        graph (dict): A dictionary to store the extracted information from the code. 
                      The keys are filenames and the values are dictionaries containing 
                      information about classes, functions, variables, imports, and function calls.
    """    
    output_file_path = Path(log_file_path).parent / "entities.txt"
    print(f"converting{log_file_path} to {output_file_path}")
    code_doc = ENTITY_EXTRACTION
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line_num = len(lines)
        entity_set = set()
        for line_idx, line in enumerate(lines):
            entity_name = ""
            entity_type = ""
            if "(name:" in line:
                entity_name=line.strip().split("name:")[1].split(")")[0].strip()
            elif "(id:" in line:
                entity_name=line.strip().split("id:")[1].split(")")[0].strip()
                if entity_name == "self":
                    continue
            elif "(arg:" in line:
                entity_name=line.strip().split("arg:")[1].split(")")[0].strip()
                entity_type="argument"
            elif "(attr:" in line:
                entity_name=line.strip().split("attr:")[1].split(")")[0].strip()
                entity_name,entity_type=check_entity_in_next_line(lines,entity_name,line_idx,line_num)
            if entity_name:
                if (entity_name,entity_type) not in entity_set:
                    entity_set.add((entity_name,entity_type))
                    if entity_type:
                        code_doc+=f'("entity"{{tuple_delimiter}}"{entity_name}"{{tuple_delimiter}}"{entity_type}"{{tuple_delimiter}}...)\n'
                    else:
                        code_doc+=f'("entity"{{tuple_delimiter}}"{entity_name}"{{tuple_delimiter}}...)\n'
    code_doc+='("relationship"{tuple_delimiter}...)\n'
    code_doc+=RELATIONSHIP_EXTRACTION
    code_doc+=json.dumps(graph[source_file], indent=4,default=list)
    code_doc=replace_file_path(code_doc,"./sourec.py")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(get_blocked_text(code_doc,"# "))

def print_ast_tree(ast_tree: ast.AST, log_file_path: str) -> None:
    """Prints and logs the structure of an Abstract Syntax Tree (AST) using the ASTPrinter class.

    Args:
        ast_tree (ast.AST): The Abstract Syntax Tree to be printed.
        log_file_path (str): The path to the log file.
    """
    printer = ASTPrinter(log_file_path)
    printer.visit(ast_tree)


def traverse_directory(root_dir: str) -> List:
    """Traverses a directory and finds all Python (.py) files. It returns a list of 
    paths to these Python files.

    Args:
        root_dir (str): The root directory to traverse.

    Returns:
        list: A list containing the paths to all Python files found in the directory.
    """
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files


def parse_file(file_path: str) -> ast.AST:
    """Parses a Python file and returns its Abstract Syntax Tree (AST).

    Args:
        file_path (str): The path to the Python file to be parsed.

    Returns:
        ast.AST: The Abstract Syntax Tree of the parsed file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()

    ast_tree = ast.parse(file_contents)
    return ast_tree


def update_graph(graph: dict, ast_tree: ast.AST, filename: str) -> None:
    """
    Updates the provided graph with information extracted from the AST of a Python file.

    This function populates the graph with details about classes, functions, variables, 
    imports, inheritance, and function calls found in the AST.

    Args:
        graph (dict): The graph structure where the extracted information is stored.
        ast_tree (ast.AST): The Abstract Syntax Tree of the Python file.
        filename (str): The name of the file being analyzed.
    """
    if filename not in graph:
        # pylint: disable=R0801
        graph[filename] = {
            'classes': set(),
            'functions': set(),
            'variables': set(),
            'imports': set(),
            'inheritance': defaultdict(set),
            'function_calls': set(),
        }
    visitor = CodeVisitor(filename, graph)
    visitor.visit(ast_tree)
    graph[filename]["function_calls"] = remove_duplicates(graph[filename]["function_calls"])


def add_node_to_node_set(dot: Digraph, node_name: str, node_set: set, node_type: str) -> set:
    """
    Adds a node to the graph visualization if it's not already present in the node set.

    Args:
        dot (Digraph): The Graphviz Digraph object to which the node is added.
        node_name (str): The name of the node to be added.
        node_set (set): A set containing the names of nodes already added to the graph.
        node_type (str): The type of the node (e.g., 'class', 'function').

    Returns:
        set: The updated set of node names after adding the new node.
    """
    if node_name not in node_set:
        dot.node(node_name, label=f'{node_name}\nType: {node_type}')
        node_set.add(node_name)
    return node_set


def add_edge_to_edge_set(dot: Digraph, edge_tuple: tuple, edge_set: set, edge_type: str) -> set:
    """
    Adds an edge to the graph visualization if it's not already present in the edge set.

    Args:
        dot (Digraph): The Graphviz Digraph object to which the edge is added.
        edge_tuple (tuple): A tuple representing the edge (source, destination).
        edge_set (set): A set containing tuples of edges already added to the graph.
        edge_type (str): The type of the edge (e.g., 'function_call').

    Returns:
        set: The updated set of edge tuples after adding the new edge.
    """
    if edge_tuple not in edge_set:
        dot.edge(edge_tuple[0], edge_tuple[1], label=f'{edge_type}')
        edge_set.add(edge_tuple)
    return edge_set


def get_list_from_chain(data: str) -> list:
    """
    Converts a chain of data in string format into a list of tuples.

    Each tuple in the list represents an element in the chain with its name and type.

    Args:
        data (str): The string representing the chain of data.

    Returns:
        list: A list of tuples with each tuple containing the name and type of an element.
    """
    if not data:
        return []
    return [(term.split(")")[-1], term.split(")")[0].split("(")[1]) for term in data.split("->")]


def build_graph_for_chain_terms(dot: Digraph, data: list, filename: str, current_node_name: str,
                                current_node_type: str, node_set: set, edge_set: set) -> Digraph:
    """
    Builds a graph for the terms in a chain by adding nodes and edges to the Digraph.

    Args:
        dot (Digraph): The Graphviz Digraph object to be updated.
        data (list): A list of tuples representing the chain terms.
        filename (str): The name of the file being analyzed.
        current_node_name (str): The name of the current node in the chain.
        current_node_type (str): The type of the current node.
        node_set (set): A set of node names already present in the graph.
        edge_set (set): A set of edge tuples already present in the graph.

    Returns:
        Digraph: The updated Graphviz Digraph object.
    """
    term_number = len(data)
    if term_number > 2:
        for term_idx in range(term_number - 2):
            upupstream_node_name = data[-3 - term_idx][0]
            upstream_node_name = upupstream_node_name + "->" + data[-2 - term_idx][0]
            upstream_node_type = data[-2 - term_idx][1]
            node_set = add_node_to_node_set(dot, upstream_node_name, node_set, upstream_node_type)
            edge_set = add_edge_to_edge_set(dot, (upstream_node_name, current_node_name), edge_set,
                                            current_node_type + "_def")
            current_node_name = copy.deepcopy(upstream_node_name)
            current_node_type = copy.deepcopy(upstream_node_type)
    else:
        edge_set = add_edge_to_edge_set(dot, (filename, current_node_name), edge_set,
                                        current_node_type + "_def")
    return dot


def remove_duplicates(function_calls: set) -> set:
    """
    Removes duplicate function calls from a set. A call is considered duplicate if it is a 
    substring of another call.

    Args:
        function_calls (set): A set of strings representing function calls.

    Returns:
        set: A set of unique function calls.
    """
    # Start with an empty list to store unique function calls
    unique_calls = []

    for call in function_calls:
        # Check if the current call is a substring of any other call
        if not any(call != other_call and call in other_call for other_call in function_calls):
            unique_calls.append(call)

    return unique_calls


def find_node_in_node_set(target_node_name: str, node_set: set) -> Union[list, None]:
    """
    Finds and returns all instances of a node name in a node set.

    Args:
        target_node_name (str): The name of the node to find.
        node_set (set): The set of nodes to search through.

    Returns:
        Union[list, None]: A list of nodes matching the target node name, or None if not found.
    """
    result_list = []
    for node in list(node_set):
        if target_node_name == node:
            result_list.append(node)
            return result_list
        node_name = node.split("->")[-1]
        if target_node_name == node_name:
            result_list.append(node)
    if result_list:
        return result_list
    return None


def output_graph(graph: dict, root_dir: str) -> None:
    """
    Outputs the given graph to a JSON file and a Graphviz diagram.

    This function writes the graph to a JSON file and creates a visual representation 
    using Graphviz, saving it as a PNG image.

    Args:
        graph (dict): The graph structure representing the Python project.
        root_dir (str): The root directory where the output files will be saved.
    """
    # JSON Output
    json_filename = os.path.join(root_dir, "codebase_graph.json")
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(graph, json_file, indent=4, default=list)
    dot = Digraph(comment='Codebase Graph')
    dot.attr('graph', nodesep='1', ranksep='4', margin='1', pad='1', sep='1')
    node_set = set()
    edge_set = set()
    for filename, data in graph.items():
        node_set = add_node_to_node_set(dot, filename, node_set, "file")
        for class_data in data['classes']:
            class_data_list = get_list_from_chain(class_data)
            upstream_node_name = class_data_list[-2][0]
            class_node_name = upstream_node_name + "->" + class_data_list[-1][0]
            node_set = add_node_to_node_set(dot, class_node_name, node_set, "class")
            dot = build_graph_for_chain_terms(dot, class_data_list, filename, class_node_name,
                                              "class", node_set, edge_set)
        for function_data in data['functions']:
            function_data_list = get_list_from_chain(function_data)
            upstream_node_name = function_data_list[-2][0]
            function_node_name = upstream_node_name + "->" + function_data_list[-1][0]
            node_set = add_node_to_node_set(dot, function_node_name, node_set, "function")
            dot = build_graph_for_chain_terms(dot, function_data_list, filename, function_node_name,
                                              "function", node_set, edge_set)
        for function_call_data in data['function_calls']:
            function_call_data_list = get_list_from_chain(function_call_data)
            if len(function_call_data_list) > 1:
                for i in range(1, len(function_call_data_list)):
                    if function_call_data_list[i][1] == "call":
                        func_call_name = function_call_data_list[i][0]
                        # func_call_node_candidate = find_node_in_node_set(func_call_name, node_set)
                        func_call_node_candidate =[func_call_name]
                        if func_call_node_candidate:
                            func_call_node_name = func_call_node_candidate[0]
                        else:
                            func_call_node_name = None
                        source_name = function_call_data_list[i - 1][0]
                        if i > 1:
                            source_upstream_name = function_call_data_list[i - 2][0]
                            source_name = source_upstream_name + "->" + source_name
                        else:
                            source_upstream_name = None
                        # source_node_candidate = find_node_in_node_set(source_name, node_set)
                        source_node_candidate=[source_name]
                        if source_node_candidate:
                            source_node_name = source_node_candidate[0]
                        else:
                            source_node_name = None
                        if func_call_node_name and source_node_name:
                            dot.edge(source_node_name, func_call_node_name, label="function_call")
                        break
            else:
                file_name = function_call_data_list[0][0]
                func_call_name = function_call_data_list[1][0]
                # func_call_node_candidate = find_node_in_node_set(func_call_name, node_set)
                func_call_node_candidate =[func_call_name]
                if func_call_node_candidate:
                    func_call_node_name = func_call_node_candidate[0]
                else:
                    func_call_node_name = None
                if func_call_node_name:
                    dot.edge(file_name, func_call_node_name, label="function_call")
    dot.render(os.path.join(root_dir, "codebase_graph"), format="png")

def get_blocked_text(text: str, indent_str: str) -> str:
    """Formats a block of text by adding an indentation string at the start of each line.

    Args:
        text (str): The text to be formatted.
        indent_str (str): The string to be added at the start of each line for indentation.

    Returns:
        str: The formatted text with added indentation.
    """     
    lines = text.split('\n')
    blocked_text=""
    for line in lines:
        blocked_text+="\n"+indent_str+line
    return blocked_text

def prepare_input_docs(root_dir: str) -> None:
    """Prepares the input documents for processing.

    This function traverses the given root directory to find all files. For each file, it parses the file into an 
    Abstract Syntax Tree (AST), logs the AST structure, and updates a graph data structure with information extracted 
    from the AST. It also extracts entities and relationships from the logged AST and updates the graph accordingly.

    Args:
        root_dir (str): The root directory containing the files to be processed.
    """
    files = traverse_directory(root_dir)
    graph = {}
    for file in files:
        log_dir=root_dir + "/" + file.split("/")[-1].split(".")[0]
        os.makedirs(log_dir, exist_ok=True)
        ast_tree = parse_file(file)
        log_file_path= f"{log_dir}/log.txt"
        print_ast_tree(ast_tree, log_file_path)
        update_graph(graph, ast_tree, file)
        extract_entities_and_relationships(log_file_path,file,graph)

@click.command()
@click.option('--root-dir',
              '-r',
              type=click.Path(exists=True),
              help='Root directory of the Python project.')
def build_graph(root_dir: str) -> None:
    """This function builds a graph representing the structure of a Python project by parsing
    the Abstract Syntax Trees (ASTs) of its files. It traverses the specified root directory,
    processes each Python file, and updates the graph with information about classes,
    functions, variables, imports, and function calls.

    Args:
        root_dir (str): The root directory of the Python project to analyze. It must be an
                        existing path.
    """
    files = traverse_directory(root_dir)
    graph = {}
    for file in files:
        log_dir=root_dir + "/" + file.split("/")[-1].split(".")[0]
        os.makedirs(log_dir, exist_ok=True)
        ast_tree = parse_file(file)
        log_file_path= f"{log_dir}/log.txt"
        print_ast_tree(ast_tree, log_file_path)
        update_graph(graph, ast_tree, file)
    output_graph(graph, root_dir)


if __name__ == '__main__':
    # pylint: disable=E1120
    build_graph()