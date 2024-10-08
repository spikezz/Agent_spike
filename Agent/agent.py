
"""
This Python module contains various utility functions and classes for interacting with AI models and handling code
blocks in text.
"""

import os
import re
import shutil
import json
import subprocess
import logging
import datetime
from pathlib import Path
from typing import Literal,Optional,Generator,Dict,Any,List,Union,Tuple

import click
import ollama
from openai import OpenAI

import graph_builder
from parquet_to_csv import convert_parquet_to_csv_func
from code_seperator import split_python_file
from prompt import (
    SYSTEM_PROMPT_DEVELOPER,
    OUTPUT_FORMAT_INSTRUCTION,
    OUTPUT_STRUCTURE_VALIDATE_INSTRUCTION,
)

def get_git_root():
    try:
        # Run the git command to get the top-level directory
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT).strip()
        return git_root.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print("Error: Not a git repository (or any of the parent directories): .git")
        return None

KNOWLEDGE_GRAPH_ROOT_PATH=f"{get_git_root()}/playground/knowledge_graph"
PLAYGROUND_ROOT_DIR=f"{get_git_root()}/playground"

def extract_count(line: str) -> Optional[int]:
    """Extracts the count from a given line of text.

    This function uses a regular expression to search for a pattern in the form of '<eval_count: number>'
    in the provided line of text. If the pattern is found, the function returns the number as an integer.
    If the pattern is not found, the function returns None.

    Args:
        line (str): The line of text from which to extract the count.

    Returns:
        int or None: The extracted count as an integer, or None if the pattern is not found.
    """ 
    pattern = re.compile(r'<eval_count:\s*(\d+)>')
    match = pattern.search(line)
    if match:
        return int(match.group(1))
    return None

def get_llm_config(llm_id: Literal["yi", "llama3"], context: str, system_prompt: str, total_eval_count: int) -> dict:
    """
    Generates a configuration dictionary for a given language model.

    Args:
        llm_id (Literal["yi", "llama3"]): The ID of the language model.
        context (str): The context for the language model to consider.
        system_prompt (str): The system prompt for the language model.
        total_eval_count (int): The total number of evaluations to perform.

    Returns:
        dict: Configuration for the specified language model.

    Raises:
        ValueError: If an unsupported llm_id is provided.
    """
    llm_configs = {
        "yi": {
            "llm": "yi:34b-chat-v1.5-q6_K",
            "num_ctx": 4096,
        },
        "llama3": {
            "llm": "llama3:70b-instruct-q5_K_M",
            "num_ctx": 8000,
        }
    }

    if llm_id not in llm_configs:
        raise ValueError(f"Unsupported language model ID: {llm_id}")

    config = llm_configs[llm_id]
    
    return {
        "user_input": context,
        "system_prompt": system_prompt,
        "num_predict": 1024,
        "total_eval_count": total_eval_count,
        **config
    }

def get_api_response(prompt: str, system_prompt: str, 
                     brand: Literal["deepseek-coder", "deepseek-chat", "gpt-4o-mini"]) -> str:
    """
    Generates a response from a specified AI model.

    Args:
        prompt (str): The user's input to which the AI model should respond.
        system_prompt (str): The system's input to which the AI model should respond.
        brand (Literal["deepseek-coder", "deepseek-chat", "gpt-4o-mini"]): The identifier for the AI model.

    Returns:
        str: The response generated by the AI model.
    """
    
    api_config = {
        "deepseek-coder": {"key": "DEEPSEEK_API_KEY", "url": "https://api.deepseek.com"},
        "deepseek-chat": {"key": "DEEPSEEK_API_KEY", "url": "https://api.deepseek.com"},
        "gpt-4o-mini": {"key": "OPENAI_API_KEY", "url": "https://api.openai.com/v1"}
    }

    config = api_config.get(brand)
    if not config:
        raise ValueError(f"Unsupported brand: {brand}")

    api_key = os.environ[config["key"]]
    client = OpenAI(api_key=api_key, base_url=config["url"])

    stream = client.chat.completions.create(
        model=brand,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            response += content
            print(content, end="", flush=True)

    return response

def get_ai_response(user_input: str, system_prompt: str, llm: str, num_ctx: int, num_predict: int,
                    total_eval_count: int) -> str:
    """Generates a response from an AI model.

    This function takes user input, a system prompt, a language model identifier, a context number, a prediction number, 
    and a total evaluation count. It generates a response from the specified AI model and returns it as a string.

    Args:
        user_input (str): The user's input to which the AI model should respond.
        system_prompt (str): The system's input to which the AI model should respond.
        llm (str): The identifier for the language model.
        num_ctx (int): The number of context tokens for the language model to consider when generating responses.
        num_predict (int): The number of tokens for the language model to predict.
        total_eval_count (int): The total number of evaluations performed so far.

    Returns:
        str: The response generated by the AI model.
    """   
    print("AI:\n", end='', flush=True)
    ai_response=""
    for text_chunk in stream_response(user_input,system_prompt,llm,num_ctx,num_predict):
        print(text_chunk, end='', flush=True)
        if text_chunk.startswith("<eval_count:"):
            total_eval_count+=extract_count(text_chunk)
        else:
            ai_response+=text_chunk
    print(f"\ntokens until now: {total_eval_count}")
    return ai_response


def stream_response(prompt: str, system_prompt: str, llm: str, num_ctx: int, 
                    num_predict: int) -> Generator[str, None, None]:
    """Generates a stream of responses from a specified AI model.

    This function takes a user prompt, a system prompt, a language model identifier, a context number, 
    and a prediction number. It generates a stream of responses from the specified AI model and yields 
    each response as a string. If 'eval_count' and 'eval_duration' are present in the response, it calculates 
    the token output rate and prints it.

    Args:
        prompt (str): The user's input to which the AI model should respond.
        system_prompt (str): The system's input to which the AI model should respond.
        llm (str): The identifier for the language model.
        num_ctx (int): The number of context tokens for the language model to consider when generating responses.
        num_predict (int): The number of tokens for the language model to predict.

    Yields:
        str: The response generated by the AI model or the string '<eval_count:eval_count>' if 'eval_count' 
             and 'eval_duration' are present in the response.
    """
    stream = ollama.generate(
        model=llm, 
        prompt=prompt, 
        system=system_prompt, 
        stream=True,
        options={
            "num_ctx":num_ctx,
            "temperature": 0.1,
            "num_predict":num_predict,
        }
    )
    for chunk in stream:
        if 'response' in chunk:
            yield chunk['response']
        if 'eval_count' in chunk and 'eval_duration' in chunk:
            eval_count = chunk['eval_count']
            eval_duration = chunk['eval_duration']
            tokens_per_second = (eval_count / eval_duration) * 1e9
            print(f"\nToken output rate: {tokens_per_second:.2f} tokens/second")
            yield f"<eval_count:{eval_count}>"

def dump_dict_to_json(path: str, dict: Dict[Any, Any]) -> None:
    """Writes a dictionary to a JSON file.

    This function takes a file path and a dictionary as arguments. It opens the file at the given path in write mode 
    and writes the dictionary to the file in JSON format. If the file does not exist, it will be created.

    Args:
        path (str): The path to the file where the dictionary should be written.
        dict (Dict[Any, Any]): The dictionary to write to the file.
    """ 
    with open(path, 'w') as f:
        json.dump(dict, f, indent=4)

def load_json(file_path: str) -> dict:
    """Loads a JSON file and returns it as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The loaded JSON file as a dictionary.
    """  
    with open(file_path, 'r') as file:
        return json.load(file)

def create_file_with_content(file_path: str, content: str) -> None:
    """Creates a new file with the given content.

    This function takes a file path and a string content as arguments. It opens the file at the given path in write
    mode and writes the content to the file. If the file does not exist, it will be created. If an error occurs
    during the process, it prints an error message.

    Args:
        file_path (str): The path where the file should be created.
        content (str): The content to be written to the file.
    """   
    try:
        # Open the file in write mode. If the file does not exist, it will be created.
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"\nFile created successfully at {file_path}")
    except Exception as e:
        print(f"\nAn error occurred while creating the file: {e}")

def is_valid_path(path: str) -> bool:
    """Checks if a given path is valid.

    This function checks if the provided path contains any invalid characters. 
    Invalid characters are '<', '>', ':', '"', '\\', '|', '?', '*'. If any of these characters are found, 
    the function returns False. Otherwise, it returns True.

    Args:
        path (str): The path to be checked for validity.

    Returns:
        bool: True if the path is valid, False otherwise.
    """  
    # Check if the path contains any invalid characters
    invalid_chars = r'<>:"\\|?*' 
    if re.search(f'[{re.escape(invalid_chars)}]', path):
        return False
    return True

def test_code_block_format(context: str) -> bool:
    """Tests if the code block format in the given context is correct.

    This function checks if the code blocks in the given context are properly formatted. 
    A code block is considered properly formatted if it starts and ends with triple backticks (```). 
    If a code block is not properly formatted, the function returns False. 
    If all code blocks are properly formatted, the function returns True.

    Args:
        context (str): The text context to check for code block formatting.

    Returns:
        bool: True if all code blocks are properly formatted, False otherwise.
    """
    pattern = re.compile(r'```(\./.*_v\d+\.\w+)')
    lines = context.split('\n')
    in_block=False
    for i, line in enumerate(lines):
        if line.startswith('```'):
            if len(line.strip()) == 3 and in_block:
                #end block
                in_block=False
            elif len(line.strip()) == 3 and not in_block:
                return False
            elif len(line.strip()) != 3:
                match = pattern.match(line)
                if match is None:
                    return False
                file_path=line.split("```")[-1]
                if is_valid_path(file_path):
                    in_block=True
                    continue
    if not in_block:
        return True
    else:
        return False


def analyse_code_block_format(context: str) -> str:
    """Analyses the format of code blocks in a given context.

    This function checks if the code blocks in the given context are properly formatted. 
    A code block is considered properly formatted if it is enclosed between triple backticks (```) 
    and if it starts with a file path followed by a version suffix after the file name. 
    If a code block is not properly formatted, the function returns an error message. 
    If all code blocks are properly formatted, the function returns a success message.

    Args:
        context (str): The context in which to analyse the code block format.

    Returns:
        str: A success message if all code blocks are properly formatted, 
             otherwise an error message indicating the issue.
    """ 
    pattern = re.compile(r'```(\./.*_v\d+\.\w+)')
    lines = context.split('\n')
    in_block=False
    for i, line in enumerate(lines):
        if line.startswith('```'):
            if len(line.strip()) == 3 and in_block:
                in_block=False
            elif len(line.strip()) == 3 and not in_block:
                return f"""
Test pattern:re.compile(r'```(\./.*_v\d+\.\w+)').match(line)
line {i+1}:start sign error, need a path following the ```
"""
            elif len(line.strip()) != 3:
                match = pattern.match(line)
                if match is None:
                    return f"""
Test pattern:re.compile(r'```(\./.*_v\d+\.\w+)').match(line)
line {i+1}: start sign error, need a path following the ``` and version suffix after the file name
"""
                file_path=line.split("```")[-1]
                if is_valid_path(file_path):
                    in_block=True
                    continue
    if not in_block:
        return "code snippet in text has correct format."
    else:
        return "Code block not closed properly before the end of the file"

class CodeBlockNotClosedError(Exception):
    """Raised when a code block in the text is not properly closed.

    This exception is raised when a code block in the text, which is supposed to be enclosed between 
    triple backticks (```), is not properly closed with triple backticks.

    Args:
        Exception (str): The error message to be displayed when the exception is raised.
    """    
    pass


def extract_code_blocks(context: str) -> List[Dict[str, Union[str, List[str]]]]:
    """Extracts code blocks from a given context.

    This function takes a string context and extracts all code blocks enclosed in triple backticks (```).
    It identifies the type of the code block based on the file extension (.py, .sh, .json, etc.) and 
    stores the code block content along with its type and file path.

    Args:
        context (str): The context from which to extract code blocks.

    Raises:
        CodeBlockNotClosedError: If a code block is not properly closed with triple backticks before the end of the file.

    Returns:
        List[Dict[str, Union[str, List[str]]]]: A list of dictionaries, each representing a code block. 
        Each dictionary contains the file path, the type of code, and the code itself.
    """   
    lines = context.split('\n')
    code_blocks = []
    in_block = False
    current_block = {}
    for i,line in enumerate(lines):
        start_match = re.match(r'```(\./.*_v\d+\.\w+)', line)
        if start_match:
            in_block = True
            if start_match.group(1).endswith('.py'):
                current_block = {
                    'file_path': start_match.group(1),
                    'type': 'python',
                    'code': [],
                }
            elif start_match.group(1).endswith('.sh'):
                current_block = {
                    'file_path': start_match.group(1),
                    'type': 'bash',
                    'code': [],
                }
            elif start_match.group(1).endswith('.json'):
                current_block = {
                    'file_path': start_match.group(1),
                    'type': 'json',
                    'code': [],
                }
            else:
                current_block = {
                    'file_path': start_match.group(1),
                    'type': 'other file',
                    'code': [],
                }
        elif in_block and re.match(r'```', line.strip()):
            in_block = False
            current_block['code'] = '\n'.join(current_block['code']).strip()
            code_blocks.append(current_block)
            current_block = {}
        elif in_block:
            current_block['code'].append(line)
    if not in_block:
        return code_blocks
    else:
        raise CodeBlockNotClosedError("Code block not closed properly before the end of the file")


def get_blocked_text(text: str, indent_str: str) -> str:
    """Prepends each line in a given text with a specified indentation string.

    Args:
        text (str): The text to be indented. It is assumed that lines in the text are separated by '\n'.
        indent_str (str): The string to be prepended to each line in the text.

    Returns:
        str: The indented text, where each line is prepended with the specified indentation string.
    """    
    lines = text.split('\n')
    blocked_text=""
    for line in lines:
        blocked_text+="\n"+indent_str+line
    return blocked_text

def get_validate_prompt(origin_ai_response: str, regex_test_result: str) -> str:
    """Generates a prompt for validation.

    This function takes the original AI response and the result of a regex test as inputs. It generates a prompt 
    for validation, which includes the original AI response, the regex test result, and an instruction for the 
    correct output format. The function returns this prompt as a string.

    Args:
        origin_ai_response (str): The original response from the AI.
        regex_test_result (str): The result of a regex test performed on the AI's response.

    Returns:
        str: A prompt for validation, which includes the original AI response, the regex test result, and an 
             instruction for the correct output format.
    """
    blocked_origin_ai_response=get_blocked_text(origin_ai_response,"\t")
    blocked_regex_test_result=get_blocked_text(regex_test_result,"\t")
    blocked_output_format_instruction=get_blocked_text(OUTPUT_FORMAT_INSTRUCTION,"\t")
    return f"""
The code blocks in last response:

{blocked_origin_ai_response}

does not pass the regex test:

{blocked_regex_test_result}

and does not follows the instruction:

{blocked_output_format_instruction}

Fix that! The response in correct format following the instruction should be:

"""

def process_validated_response(role: str, user_input: str, ai_response: str, context: str, 
                               chat_history: Dict[str, List[Dict[str, str]]], 
                               code_blocks_traj: List[Dict[str, List[Dict[str, str]]]]) -> str:
    """Processes the validated response from the AI model and updates the context, chat history, and code blocks
    trajectory.

    Args:
        role (str): The role of the user in the chat (e.g., 'user', 'system').
        user_input (str): The input provided by the user to the AI model.
        ai_response (str): The response generated by the AI model.
        context (str): The current context of the conversation.
        chat_history (Dict[str, List[Dict[str, str]]]): The history of the chat, stored as a dictionary where
        the key is 'chat_history' and the value is a list of dictionaries, each containing 'user' and 'AI'
        keys with corresponding messages.
        code_blocks_traj (List[Dict[str, List[Dict[str, str]]]]): The trajectory of code blocks generated
        during the conversation. Each element in the list is a dictionary with the key 'code_block' and the
        value is a list of dictionaries representing individual code blocks.

    Returns:
        str: The updated context of the conversation.
    """
    if role:
        context+=f"\n{role}:\n{user_input}\n"
    context+=f"\nAI: \n{ai_response}\n"
    chat_history["chat_history"].append({"user":user_input,"AI":ai_response})
    current_code_blocks = extract_code_blocks(ai_response)
    os.makedirs(f"{PLAYGROUND_ROOT_DIR}/output/", exist_ok=True)
    for code_block in current_code_blocks:
        create_file_with_content(os.path.join(f"{PLAYGROUND_ROOT_DIR}/output/", os.path.basename(code_block['file_path'])),code_block['code'])
    code_blocks_traj.append({"code_block":current_code_blocks})
    return context

def wrap_file_content(input_dir: str) -> str:
    """Wraps the content of all Python files in a directory with markdown code blocks.

    This function takes a directory path as input, reads all Python files in that directory, 
    and wraps their content with markdown code blocks. The resulting string is returned.

    Args:
        input_dir (str): The path to the directory containing the Python files.

    Returns:
        str: A string containing the content of all Python files in the directory, each wrapped with markdown code
        blocks.
    """     
    result=""
    for filename in os.listdir(input_dir):
        if filename.endswith('.py'):
            file_path = os.path.join(input_dir, filename)
            result+=f"```./input/{filename}\n"
            with open(file_path, 'r') as infile:
                result+=infile.read()
            result+="\n```\n\n"
    return result

def append_graph_context(root_dir: str) -> None:
    """Appends the context of a knowledge graph.

    This function walks through the root directory and for each file, it checks if there's a directory with the same
    stem. If such a directory exists, it concatenates the contents of the file and the 'entities.txt' file in the
    directory, and writes the result to a new file in the 'knowledge_graph/input/' directory. The directory is then
    removed.

    Args:
        root_dir (str): The root directory to walk through.
    """       
    os.makedirs(f"{PLAYGROUND_ROOT_DIR}/knowledge_graph/input/", exist_ok=True)
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_stem=Path(file).stem
            for dir in dirs:
                if dir==file_stem:
                    print(f"enter:{dir}")
                    concatenate_files(
                        f"{PLAYGROUND_ROOT_DIR}/output/{file}", f"{PLAYGROUND_ROOT_DIR}/output/{dir}/entities.txt",
                        f"{PLAYGROUND_ROOT_DIR}/knowledge_graph/input/{file_stem}.txt"
                    )
                    remove_directory(f"{PLAYGROUND_ROOT_DIR}/output/{dir}/")

def concatenate_files(file1_path: str, file2_path: str, output_path: str) -> None:
    """Concatenates the contents of two files and writes the result to a new file.

    This function reads the contents of the first and second input files, concatenates them, 
    and writes the result to the output file. If an IOError occurs during this process, 
    the function prints an error message.

    Args:
        file1_path (str): The path to the first input file.
        file2_path (str): The path to the second input file.
        output_path (str): The path to the output file where the concatenated contents will be written.
    """     
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output_file:
            # Read and write contents of the first file
            output_file.write("")
            output_file.write(file1.read())
            # Add a newline between files if needed
            output_file.write('\n')
            # Read and write contents of the second file
            output_file.write(file2.read())
            output_file.write('\n')
        print(f"Files concatenated successfully. Output saved to {output_path}")
    except IOError as e:
        print(f"An error occurred: {e}")

def remove_directory(directory: str) -> None:
    """Removes a specified directory and all its contents.

    This function checks if the given directory exists. If it does, it attempts to remove the directory 
    and all its contents. If the directory does not exist, or if an error occurs during removal, 
    it prints an appropriate message.

    Args:
        directory (str): The path to the directory to be removed.
    """      
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    try:
        # Remove the entire directory and its contents
        shutil.rmtree(directory)
        print(f"Removed directory and all its contents: {directory}")
    except Exception as e:
        print(f"Failed to remove {directory}. Reason: {e}")

def execute_graphrag_bash_command(python_path: str, module: str, root_path: str, init: bool = False,
                                  query_type: Optional[str] = None, 
                                  query: Optional[str] = None) -> Tuple[subprocess.Popen, Optional[str]]:
    """Executes a GraphRAG bash command.

    This function executes a GraphRAG bash command using the specified Python path, module, and root path. 
    It can optionally initialize the GraphRAG index, and perform a query of a specified type.

    Args:
        python_path (str): The path to the Python interpreter.
        module (str): The GraphRAG module to be executed.
        root_path (str): The root path for the GraphRAG index.
        init (bool, optional): Whether to initialize the GraphRAG index. Defaults to False.
        query_type (str, optional): The type of query to perform, if any. Defaults to None.
        query (str, optional): The query to perform, if any. Defaults to None.

    Raises:
        subprocess.CalledProcessError: If the subprocess call returns a non-zero exit code.

    Returns:
        Tuple[subprocess.Popen, Optional[str]]: The subprocess.Popen object representing the running process, 
        and None (since we don't have direct access to stdout here).
    """   
    if init:
        cmd = [python_path, "-m", "graphrag.index", "--init" ,"--root", root_path]
        log_file=f"{PLAYGROUND_ROOT_DIR}/graph_building_log.log"
    elif module=="graphrag.query":
        cmd = [python_path, "-m", "graphrag.query", "--root", root_path, "--method",query_type, f"\"{query}\""]
        log_file=f"{PLAYGROUND_ROOT_DIR}/graph_query_log.log"
    else:
        cmd = [python_path, "-m", "graphrag.index", "--root", root_path]
        log_file=f"{PLAYGROUND_ROOT_DIR}/graph_building_log.log"
    print(f"running cmd: {cmd}")
    try:
        with open(log_file, 'a') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            show_in_terminal=False
            for line in process.stdout:
                if line.startswith("SUCCESS:"):
                    show_in_terminal=True
                if module=="graphrag.query" and show_in_terminal:
                    print(line,"\n")
                log.write(line)  # Write to log file
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        logging.info("Command executed successfully.")
        return process, None  # We don't have direct access to stdout here
    except FileNotFoundError as e:
        logging.error(f"Error: {str(e)}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command. Return code: {e.returncode}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

def find_directory(root_dir, target_dir_name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_dir_name in dirnames:
            return os.path.join(dirpath, target_dir_name)
    return None

def find_file(root_dir, target_file_name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_file_name in filenames:
            return os.path.join(dirpath, target_file_name)
    return None

def replace_file(root_dir):
    # Find source directories and files
    source_prompts_dir = find_directory(root_dir, 'code_explainer_prompts')
    source_settings_file = find_file(root_dir, 'settigns.yaml')

    if not source_prompts_dir or not source_settings_file:
        print('Required directories or files not found.')
        return

    # Define destination directories and files
    destination_prompts_dir = os.path.join(root_dir, f'{PLAYGROUND_ROOT_DIR}/knowledge_graph/prompts')
    destination_settings_file = os.path.join(root_dir, f'{PLAYGROUND_ROOT_DIR}/knowledge_graph/settings.yaml')

    # Ensure the destination directories exist
    os.makedirs(destination_prompts_dir, exist_ok=True)

    # Copy prompt files
    for filename in os.listdir(source_prompts_dir):
        source_file = os.path.join(source_prompts_dir, filename)
        destination_file = os.path.join(destination_prompts_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)
            print(f'Copied {source_file} to {destination_file}')

    # Copy settings file
    shutil.copy2(source_settings_file, destination_settings_file)
    print(f'Copied {source_settings_file} to {destination_settings_file}')

@click.command()
@click.option('--llm', default="gpt-4o-mini", help='llm model.')
@click.option('--graph-ready', '-gr', is_flag=True, help='Indicates if it is ready to query knowledge graph.')
@click.option("--python-path", type=str,default="python3")
def main(llm:str,graph_ready:bool,python_path:str) -> None:
    """Main function to interact with the AI model and knowledge graph.

    This function initiates a chat session with the AI model. It supports various commands to interact with the
    knowledge graph, such as querying the graph, indexing the graph, and preparing the graph for queries. It
    also handles the AI model's responses, validates the output format, and saves the chat history and code
    blocks generated during the session.

    Args:
        llm (str): The ID of the language model to use. This should be either 'gpt-4o-mini', 'deepseek-coder',
        or 'deepseek-chat'.
        graph_ready (bool): A flag indicating whether the knowledge graph is ready for querying.
        python_path (str): The path to the Python interpreter to use for executing commands related to the
        knowledge graph.
    """ 
    code_blocks_traj=[]
    chat_history={"chat_history":[]}
    context=""
    turn_idx=0
    validator_triggered=0
    while True:
        print(f"Chat round {turn_idx}:")
        user_input = input("user: ")
        if user_input.lower() == '/q':
            print("Goodbye!")
            break
        elif user_input.lower() == '/pcg':
            print(f"user:\n{user_input}\n")
            graph_ready=False
            split_python_file(f"{PLAYGROUND_ROOT_DIR}/input",f"{PLAYGROUND_ROOT_DIR}/output")
            continue
        elif user_input.lower() == '/cg':
            print(f"user:\n{user_input}\n")
            graph_builder.prepare_input_docs(f"{PLAYGROUND_ROOT_DIR}/output")
            append_graph_context(f"{PLAYGROUND_ROOT_DIR}/output")
            continue
        elif user_input.lower() == '/itg':
            print(f"user:\n{user_input}\n")
            graph_ready=False    
            result,output = execute_graphrag_bash_command(python_path,
                                           "graphrag.index",
                                           KNOWLEDGE_GRAPH_ROOT_PATH,
                                           init=True
            )
            if result:
                print("Command execution completed successfully.")
            else:
                print("Command execution failed.")
            replace_file(get_git_root())
            print("Please copy the API key to .env by needs.")
            continue
        elif user_input.lower() == '/ixg':
            print(f"user:\n{user_input}\n")
            result,output = execute_graphrag_bash_command(python_path,
                                           "graphrag.index",
                                           KNOWLEDGE_GRAPH_ROOT_PATH,
            )
            if result:
                print("Command execution completed successfully.")
            else:
                print("Command execution failed.")
            csv_import_folder_path=KNOWLEDGE_GRAPH_ROOT_PATH+"/import"
            os.makedirs(csv_import_folder_path, exist_ok=True)
            output_folders=[]
            for item in os.listdir(KNOWLEDGE_GRAPH_ROOT_PATH+"/output"):
                full_path = os.path.join(KNOWLEDGE_GRAPH_ROOT_PATH, "output", item)
                if os.path.isdir(full_path):
                    creation_time = os.path.getctime(full_path)
                    output_folders.append((datetime.datetime.fromtimestamp(creation_time),full_path))
            last_output_dir_path=sorted(output_folders, key=lambda x: x[0],reverse=True)[0][1]+"/artifacts"
            convert_parquet_to_csv_func(last_output_dir_path,csv_import_folder_path)
            graph_ready=True
            continue
        elif graph_ready and user_input.lower() == '/qg':
            print(f"user:\n{user_input}\n")
            user_input = input("user: ")
            # user_input=""
            query_type="local"
            if user_input.lower() == '/g':
                print(f"user:\n{user_input}\n")
                query_type="global"
                user_input = input("user: ")
            print(f"user:\n{user_input}\n")
            result,output = execute_graphrag_bash_command(python_path,
                                           "graphrag.query",
                                           KNOWLEDGE_GRAPH_ROOT_PATH,
                                           query_type=query_type,
                                           query=user_input
            )
            if result:
                print("query response:")
                print(output)
            continue
        context+=f"\nuser:\n{user_input}\n"
        # Response
        validate_response=False
        system_prompt=SYSTEM_PROMPT_DEVELOPER+OUTPUT_FORMAT_INSTRUCTION
        for retry in range(0,2):
            print(f"retry:{retry}")
            origin_ai_response=get_api_response(context,system_prompt,llm)
            if test_code_block_format(origin_ai_response):
                context=process_validated_response(None,user_input,origin_ai_response,context,chat_history,code_blocks_traj)
                validate_response=True
                break
            else:
                validator_triggered+=1
                regex_test_result=analyse_code_block_format(origin_ai_response)
                validate_prompt=get_validate_prompt(origin_ai_response,regex_test_result)
                print(f"AI_validator:\n{validate_prompt}\n")
                validate_ai_response=get_api_response(validate_prompt,OUTPUT_STRUCTURE_VALIDATE_INSTRUCTION,"gpt-4o-mini")
                if test_code_block_format(validate_ai_response):
                    context=process_validated_response("AI_validator",validate_prompt,validate_ai_response,context,chat_history,code_blocks_traj)
                    validate_response=True
                    break
        if not validate_response:
            raise "Bad output!"
        dump_dict_to_json(f"{PLAYGROUND_ROOT_DIR}/code_blocks.json",{"code_blocks_traj":code_blocks_traj, "validator_triggered":validator_triggered})
        dump_dict_to_json(f"{PLAYGROUND_ROOT_DIR}/chat_history.json",chat_history)
        # Dump context
        create_file_with_content(f"{PLAYGROUND_ROOT_DIR}/context.log",context)
        turn_idx+=1

if __name__ == "__main__":
    main()
