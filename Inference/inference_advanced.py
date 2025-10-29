# Mapping guidance: how Engine/ModelName from YAML maps to API usage in this runner.
# - Engine 'qwen': uses an OpenAI-compatible client configured with qwen_api_key/qwen_url (DashScope).
# - Engine 'deerapi_gpt': uses an OpenAI-compatible client configured with deerapi_key/deerapi_url (proxy).
#   Some ModelName values (e.g., grok/claude/gemini listed below) are routed with special handling.
# To add a new provider, extend client_init(engine_type) and add model-specific branches in eval_message if needed.
import random
import concurrent.futures
import argparse
import os
import json
import sys
import yaml
from tools.load_config_tools import load_config
import re
from zhipuai import ZhipuAI
import http.client
from urllib.parse import urlparse
from openai import AzureOpenAI, OpenAI
import base64
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional
from pydantic import RootModel
import base64
import json
import ray
import time
import os
import random
from tools.message_gen_tools import generate_message

# New imports for robustness & logging
import traceback
import threading
import shutil
from datetime import datetime

# Global lock for error.jsonl concurrent writes
ERROR_LOG_LOCK = threading.Lock()


def _truncate_text(text, limit=8000):
    try:
        s = str(text)
    except Exception:
        s = repr(text)
    if s is None:
        return None
    return s if len(s) <= limit else s[:limit] + "...<truncated>"


def _build_failed_result(question_id, message_debug=None, err=None, raw_response=None, is_success=False):
    temp = {
        "id": question_id,
        "is_success": is_success,
        "result": None,
    }
    if message_debug is not None:
        temp["messages"] = message_debug
    temp["response"] = {
        "error": _truncate_text(err) if err else None,
        "raw": _truncate_text(raw_response) if raw_response is not None else None,
    }
    return temp


def _extract_json_from_content_text(content: str):
    # Try to extract a JSON object from the text. Fallback to simple thought/answer regex.
    if not content:
        return {"thought": None, "answer": None}

    # 1) Handle in-content function call with tags
    try:
        m = re.search(r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>", content, re.DOTALL)
        if m:
            blob = m.group(1).strip()
            try:
                data = json.loads(blob)
                # normalize to dict
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    params = data.get("parameters")
                    if params is None:
                        func = data.get("function") or {}
                        params = func.get("arguments")
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except Exception:
                            params = None
                    if isinstance(params, dict):
                        ans = params.get("answer")
                        th = params.get("thought")
                        if ans is not None:
                            return {"thought": th, "answer": ans}
            except Exception:
                pass
    except Exception:
        pass

    # 2) Handle array form without tags
    try:
        stripped = content.strip()
        if stripped.startswith("["):
            arr = json.loads(stripped)
            if isinstance(arr, list) and arr:
                data = arr[0]
                if isinstance(data, dict):
                    params = data.get("parameters")
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except Exception:
                            params = None
                    if isinstance(params, dict):
                        ans = params.get("answer")
                        th = params.get("thought")
                        if ans is not None:
                            return {"thought": th, "answer": ans}
    except Exception:
        pass

    # 3) Original: extract first JSON object
    try:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            # Handle {"parameters": {...}} shape
            if isinstance(obj, dict) and "answer" not in obj:
                params = obj.get("parameters")
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = None
                if isinstance(params, dict) and "answer" in params:
                    return {"thought": params.get("thought"), "answer": params.get("answer")}
            if isinstance(obj, dict) and "answer" in obj:
                return obj
    except Exception:
        pass

    # 4) Original strict JSON regex for {"thought": "...", "answer": "..."}
    m = re.search(
        r'\{\s*"thought"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*\}',
        content,
        re.DOTALL,
    )
    if m:
        return {"thought": m.group(1), "answer": m.group(2)}

    return {"thought": None, "answer": None}


def _extract_response_json_from_openai_like(response_obj):
    # Compatible with SDK response objects (OpenAI/Azure-like)
    try:
        choice = response_obj.choices[0]
        # tool_calls first
        tcalls = getattr(choice.message, "tool_calls", None)
        if tcalls:
            try:
                args = tcalls[0].function.arguments
                return json.loads(args)
            except Exception:
                pass
        # function_call next
        fcall = getattr(choice.message, "function_call", None)
        if fcall and getattr(fcall, "arguments", None):
            try:
                return json.loads(fcall.arguments)
            except Exception:
                pass
        # content fallback
        content = getattr(choice.message, "content", "") or ""
        return _extract_json_from_content_text(content)
    except Exception:
        return {"thought": None, "answer": None}


def _extract_response_json_from_deerapi_dict(data: dict):
    # Compatible with raw dict responses (e.g., deerapi proxy)
    try:
        msg = data["choices"][0].get("message", {})
        tcalls = msg.get("tool_calls")
        if tcalls:
            try:
                args = tcalls[0]["function"]["arguments"]
                return json.loads(args)
            except Exception:
                pass
        fcall = msg.get("function_call")
        if fcall and "arguments" in fcall:
            try:
                return json.loads(fcall["arguments"])
            except Exception:
                pass
        content = msg.get("content", "") or ""
        return _extract_json_from_content_text(content)
    except Exception:
        return {"thought": None, "answer": None}


def init_log_dir(yaml_path, config_obj):
    # Create ./log/{timestamp}/ with config artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("log", ts)
    os.makedirs(log_dir, exist_ok=True)

    # Copy original yaml into log_dir/config.yaml
    try:
        if yaml_path and os.path.exists(yaml_path):
            shutil.copy2(yaml_path, os.path.join(log_dir, "config.yaml"))
    except Exception:
        # Best-effort; do not crash
        pass

    # Dump parsed config into config.json
    try:
        with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as cf:
            json.dump(config_obj, cf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # Prepare failed_jsons dir
    os.makedirs(os.path.join(log_dir, "failed_jsons"), exist_ok=True)

    return log_dir


def write_error_log(log_dir, error_record):
    if not log_dir:
        return
    error_path = os.path.join(log_dir, "error.jsonl")
    line = json.dumps(error_record, ensure_ascii=False)
    with ERROR_LOG_LOCK:
        with open(error_path, "a", encoding="utf-8") as ef:
            ef.write(line + "\n")


def copy_failed_json(log_dir, src_path):
    if not log_dir or not src_path:
        return
    dst_dir = os.path.join(log_dir, "failed_jsons")
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src_path)
    dst = os.path.join(dst_dir, base)
    if os.path.abspath(src_path) == os.path.abspath(dst):
        return
    # If conflict, add suffix
    if os.path.exists(dst):
        name, ext = os.path.splitext(base)
        i = 1
        while True:
            cand = os.path.join(dst_dir, f"{name}_{i}{ext}")
            if not os.path.exists(cand):
                dst = cand
                break
            i += 1
    try:
        shutil.copy2(src_path, dst)
    except Exception:
        # Best-effort; ignore
        pass


# Engine routing note:
# - 'qwen': OpenAI-compatible client using qwen_api_key/qwen_url (DashScope).
# - 'deerapi_gpt': OpenAI-compatible client using deerapi_key/deerapi_url (proxy).
# Other engines in example YAMLs (e.g., 'claude', 'gemini', 'doubao') are not implemented here.
# Use 'deerapi_gpt' with appropriate ModelName via proxy, or extend client_init as documented in README.
def client_init(engine_type):
    load_dotenv(dotenv_path="../.env")
    api_key = os.getenv("deerapi_key")
    base_url = os.getenv("deerapi_url")
    qwen_api_key = os.getenv("qwen_api_key")
    qwen_url = os.getenv("qwen_url")
    
    if engine_type == "qwen":
        # Do not modify qwen URL usage as per requirement
        client = OpenAI(
            api_key=qwen_api_key,
            base_url=qwen_url,
        )
        return client
    if engine_type == "deerapi_gpt"  :
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return client


class ChoiceAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class ThoughtChoiceAnswer(BaseModel):
    thought: str
    answer: Literal["A", "B", "C", "D"]


class ThoughtChoiceMoreAnswer(BaseModel):
    thought: str
    answer: Literal["A", "B", "C", "D", "E", "F", "G"]


class YesOrNoAnswer(BaseModel):
    answer: Literal["yes", "no", "unknown"]


class ThoughtYesOrNoAnswer(BaseModel):
    thought: str
    answer: Literal["yes", "no", "unknown"]


answer_multiple_choice_questions_functions = [
    {
        "name": "answer_multiple_choice_questions",
        "description": "Generate the answer option for multiple choice questions. ",
        "parameters": ChoiceAnswer.model_json_schema(),
    }
]

answer_multiple_choice_questions_withThought_functions = [
    {
        "name": "answer_multiple_choice_questions_withThought_functions",
        "description": "Generate thought and corresponding answer option for multiple choice questions. ",
        "parameters": ThoughtChoiceAnswer.model_json_schema(),
    }
]

answer_multiple_choice_questions_more_withThought_functions = [
    {
        "name": "answer_multiple_choice_questions_more_withThought_functions",
        "description": "Generate thought and corresponding answer option for multiple choice questions. ",
        "parameters": ThoughtChoiceMoreAnswer.model_json_schema(),
    }
]

answer_yes_or_no_questions_functions = [
    {
        "name": "answer_yes_or_no_questions",
        "description": "Generate the answer option for yes or no questions.",
        "parameters": YesOrNoAnswer.model_json_schema(),
    }
]

answer_yes_or_no_questions_withThought_functions = [
    {
        "name": "answer_yes_or_no_questions_withThought",
        "description": "Generate thought and corresponding answer option for yes or no questions.",
        "parameters": ThoughtYesOrNoAnswer.model_json_schema(),
    }
]


def convert_to_responses_message(messages):
    responses_api = []

    for msg in messages:
        response_msg = {
            "type": "message",
            "role": msg.get("role", "user"),
            "content": [],
        }

        for item in msg.get("content", []):
            # If text content
            if item.get("type") == "text":
                response_msg["content"].append(
                    {"type": "text", "text": item.get("text", "")}
                )
            # If image content
            elif item.get("type") == "image":
                # image_url can be a raw string or a dict {'url': '...'}
                img_url = item.get("image_url")
                response_msg["content"].append({"type": "image", "image_url": img_url})

        responses_api.append(response_msg)

    return responses_api


def eval_message(client,engine_type, model, json_data, enable_thinking, kwargs, ablation_options):
    question_id = json_data.get("question_id")
    question_type = json_data.get("question_type")
    try:
        messages, option_mappings, message_debug = generate_message(
            json_data, enable_thinking, ablation_options
        )
    except Exception:
        return _build_failed_result(
            question_id,
            None,
            f"generate_message error: {traceback.format_exc()}",
            None,
            False,
        )

    if not messages:
        return _build_failed_result(
            question_id, None, "empty messages from generate_message", None, False
        )

    # Validate images/content existence more safely
    try:
        second_msg_content = []
        if len(messages) > 1 and isinstance(messages[1], dict):
            second_msg_content = messages[1].get("content", []) or []
        for item in second_msg_content:
            if item is None:
                return _build_failed_result(
                    question_id,
                    "image dir error.",
                    "invalid image item None",
                    None,
                    False,
                )
    except Exception:
        return _build_failed_result(
            question_id,
            None,
            f"messages structure error: {traceback.format_exc()}",
            None,
            False,
        )

    # Select functions according to question_type and enable_thinking
    option_length = len(json_data.get("option_text", [])) if isinstance(json_data.get("option_text", []), list) else 0
    if question_type == "yes_or_no":
        functions = (
            answer_yes_or_no_questions_withThought_functions
            if enable_thinking
            else answer_yes_or_no_questions_functions
        )
    elif question_type == "multiple_choice":
        if enable_thinking:
            if option_length == 4:
                functions = answer_multiple_choice_questions_withThought_functions
            else:
                functions = answer_multiple_choice_questions_more_withThought_functions
        else:
            functions = answer_multiple_choice_questions_functions
    else:
        # Not implemented question type; return failure
        return _build_failed_result(
            question_id,
            message_debug,
            f"not implemented for question type: {question_type}",
            None,
            False,
        )

    max_retries = 3
    is_success = False
    response_json = {"thought": None, "answer": None}
    response_raw = None
    last_error = None

    # Ensure timeout present
    local_kwargs = dict(kwargs or {})
    local_kwargs.setdefault("timeout", 120)

    for attempt in range(1, max_retries + 1):
        try:
            # Special routing note:
            # For certain ModelName values (grok/claude/gemini), we call an OpenAI-compatible proxy endpoint.
            # If you add a new model that requires a different path, extend this branch.
            # deerapi proxy via http.client for certain model names
            if model in ["grok-4-0709", "claude-sonnet-4-20250514","gemini-2.5-pro-preview-06-05"]:
                api_key = os.getenv("deerapi_key")
                base_url = os.getenv("deerapi_url", "")
                u = urlparse(base_url or "")
                host = u.netloc or "api.deerapi.com"
                base_path = u.path.rstrip("/") if u.path else "/v1"
                endpoint = f"{base_path}/chat/completions"
                payload = json.dumps(
                    {
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "tools": functions,
                        "tool_choice": functions[0]["name"],
                        **local_kwargs,
                    }
                )
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                conn = http.client.HTTPSConnection(host) if (u.scheme or "https") == "https" else http.client.HTTPConnection(host)
                conn.request("POST", endpoint, body=payload, headers=headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                response_raw = data
                deer_json = json.loads(data)
                response_json = _extract_response_json_from_deerapi_dict(deer_json)
                if response_json.get("answer"):
                    is_success = True
                    response = deer_json  # for original_response compatibility
                    break
                raise RuntimeError("no valid answer extracted from deerapi response")

            elif model in ["doubao-1-5-thinking-vision-pro-250428"]:
                client = client_init(engine_type)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[{"type": "function", "function": functions[0]}],
                    tool_choice={
                        "type": "function",
                        "function": {"name": functions[0]["name"]},
                    },
                    **local_kwargs,
                )
                response_raw = _truncate_text(response)
                response_json = _extract_response_json_from_openai_like(response)
                if response_json.get("answer"):
                    is_success = True
                    break
                raise RuntimeError("no valid answer extracted from response")

            else:
                client = client_init(engine_type)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[{"type": "function", "function": functions[0]}],
                    tool_choice={
                        "type": "function",
                        "function": {"name": functions[0]["name"]},
                    },
                    **local_kwargs,
                )
                response_raw = _truncate_text(response)
                response_json = _extract_response_json_from_openai_like(response)
                if response_json.get("answer"):
                    is_success = True
                    break
                raise RuntimeError("no valid answer extracted from response")
        except Exception:
            last_error = f"[Attempt {attempt}] {traceback.format_exc()}"
            # Exponential backoff with jitter
            sleep_s = min(2 ** (attempt - 1) * 2 + random.uniform(0, 1), 30)
            time.sleep(sleep_s)

    # Unified output
    if response_json.get("answer"):
        temp = {
            "id": question_id,
            "answer": response_json.get("answer", ""),
            "thought": response_json.get("thought", ""),
            "is_success": is_success,
            "messages": message_debug,
            "original_response": _truncate_text(response_raw),
            "response": {"raw": _truncate_text(response_raw), "error": None},
        }
        return temp
    else:
        return _build_failed_result(
            question_id, message_debug, last_error or "no answer", response_raw, False
        )


def eval_response_message(client, model, json_data, enable_thinking, kwargs, visual_prompt=True):
    question_id = json_data.get("question_id")
    question_type = json_data.get("question_type")
    try:
        messages, option_mappings, message_debug = generate_message(
            json_data, enable_thinking, visual_prompt
        )
    except Exception:
        return _build_failed_result(
            question_id,
            None,
            f"generate_message error: {traceback.format_exc()}",
            None,
            False,
        )

    # Validate images/content existence
    try:
        second_msg_content = []
        if len(messages) > 1 and isinstance(messages[1], dict):
            second_msg_content = messages[1].get("content", []) or []
        for item in second_msg_content:
            if item is None:
                return _build_failed_result(
                    question_id,
                    "image dir error.",
                    "invalid image item None",
                    None,
                    False,
                )
    except Exception:
        return _build_failed_result(
            question_id,
            None,
            f"messages structure error: {traceback.format_exc()}",
            None,
            False,
        )

    responses_messages = convert_to_responses_message(messages)

    if question_type == "yes_or_no":
        functions = (
            answer_yes_or_no_questions_withThought_functions
            if enable_thinking
            else answer_yes_or_no_questions_functions
        )
    elif question_type == "multiple_choice":
        functions = (
            answer_multiple_choice_questions_withThought_functions
            if enable_thinking
            else answer_multiple_choice_questions_functions
        )
    else:
        return _build_failed_result(
            question_id,
            message_debug,
            f"not implemented for question type: {question_type}",
            None,
            False,
        )

    max_retries = 3
    is_success = False
    response_json = {"thought": None, "answer": None}
    response_raw = None
    last_error = None

    local_kwargs = dict(kwargs or {})
    local_kwargs.setdefault("timeout", 60)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=responses_messages,
                reasoning={"effort": "minimal"},
                **local_kwargs,
            )
            response_raw = _truncate_text(response)
            response_json = _extract_response_json_from_openai_like(response)
            if response_json.get("answer"):
                is_success = True
                break
            raise RuntimeError("no valid answer extracted from response")
        except Exception:
            last_error = f"[Attempt {attempt}] {traceback.format_exc()}"
            sleep_s = min(2 ** (attempt - 1) * 2 + random.uniform(0, 1), 30)
            time.sleep(sleep_s)

    if response_json.get("answer"):
        temp = {
            "id": question_id,
            "answer": response_json.get("answer", ""),
            "thought": response_json.get("thought", ""),
            "is_success": is_success,
            "messages": message_debug,
            "original_response": _truncate_text(response_raw),
            "response": {"raw": _truncate_text(response_raw), "error": None},
        }
        return temp
    else:
        return _build_failed_result(
            question_id, message_debug, last_error or "no answer", response_raw, False
        )


def worker(task_item, index, engine_type, model_type, thinking, result_file_dir, kwargs, ablation_options, log_dir=None, yaml_path=None):
    with open(task_item, "r", encoding="utf-8") as f:
        json_data = json.load(f)  #

    client = client_init(engine_type)

    # Run evaluation
    result_dict = eval_message(client,engine_type, model_type, json_data, thinking, kwargs, ablation_options)

    # Ensure results dir exists and save
    os.makedirs(result_file_dir, exist_ok=True)
    save_file_path = os.path.join(result_file_dir, os.path.basename(task_item))
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    # On failure, append to error.jsonl and copy failing json to log_dir/failed_jsons
    if not result_dict.get("is_success", False):
        error_record = {
            "id": result_dict.get("id"),
            "engine_type": engine_type,
            "model_type": model_type,
            "yaml_path": yaml_path,
            "json_path": task_item,
            "result_file_path": save_file_path,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "response_error": (result_dict.get("response") or {}).get("error"),
            "response_raw": (result_dict.get("response") or {}).get("raw"),
        }
        write_error_log(log_dir, error_record)
        copy_failed_json(log_dir, task_item)

    return result_dict


def run_single(config, json_file_list, num_workers=5, log_dir=None, yaml_path=None):
    max_workers = min(num_workers, len(json_file_list))
    if max_workers == 0:
        print("No need to run.")
        return
    engine = config["Engine"]
    model = config["ModelName"]
    thinking = config["thinking"]
    ResultsDir = config["ResultsDir"]
    ablation_options = config["ablation_options"]

    result_file_dir = os.path.join(ResultsDir, model)
    process = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                worker,
                tm,
                idx,
                engine,
                model,
                thinking,
                result_file_dir,
                config["kwargs"],
                ablation_options,
                log_dir,
                yaml_path,
            ): idx
            for idx, tm in enumerate(json_file_list)
        }
        for f in concurrent.futures.as_completed(futures):
            process.append(f.result())
            print(f"{model} Processed {len(process)}/{len(json_file_list)} tasks")
    print(f"All tasks processed.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_dir",
        default=None,
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config, json_file_list = load_config(args.yaml_dir)
    # Initialize logging directory and persist configs
    log_dir = init_log_dir(args.yaml_dir, config)
    # Execute
    run_single(config, json_file_list, config["MaxWorker"], log_dir=log_dir, yaml_path=args.yaml_dir)


if __name__ == "__main__":
    main()

# example cases 
# python inference.py --yaml_dir configs/ModelConfigList/
