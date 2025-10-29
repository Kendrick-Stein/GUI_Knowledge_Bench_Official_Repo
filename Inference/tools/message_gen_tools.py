# pylint: disable=no-member
import random
from typing import Dict, Any
import base64
from dotenv import load_dotenv
import random
from typing import List, Optional, Union, Dict, Any
import json 
import os 
from pprint import pprint
import cv2
from tools.draw_picture import draw_action, _ensure_image_exists,draw_annotation
import hashlib

def find_image_path(image_path):
    load_dotenv(dotenv_path='../.env') 
    
    dataset_base_dir = os.getenv('dataset_base_dir', '[]')
    try:
        base_dirs = json.loads(dataset_base_dir)
    except json.JSONDecodeError:
        print("dataset_base_dir has invalid JSON format; please check .env")
        return None
    
    for base in base_dirs:
        candidate_path = os.path.join(base, image_path)
        if os.path.exists(candidate_path):
            return candidate_path
    if os.path.exists(image_path):
        return image_path
    
    print(f"Image not found: {image_path}")
    print("Tried paths:", [os.path.join(base, image_path) for base in base_dirs if base])
    return None


def get_image_size(image_path):
    image_path = find_image_path(image_path)
    if image_path is None:
        return None
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
            return f'The size of the image is {w}x{h}.\n'
            # return {"width": int(w), "height": int(h)}
    except Exception as e:
        print(f"Error loading image: {e}")
        return ''

    
def encode_image(image_path):
    image_path = find_image_path(image_path)
    if image_path is None:
        return None
    # Determine the MIME type based on the file 
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        mime_type = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".gif":
        mime_type = "image/gif"
    else:
        mime_type = "application/octet-stream"  # default for other types
        
    with open(image_path, "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode('utf-8')
        
    base64_url = f"data:{mime_type};base64,{encoded_img}"
    image_message = {
        "type": "image_url",
        "image_url": {"url": base64_url},
        "detail": "high"
    }
    return image_message


# ===================== Options Generation =====================
def _process_options(q_id: str,
                     q_type: str,
                     option_texts: Optional[List[str]],
                     option_images: Optional[List[str]]) -> (List[Dict], Dict[int, int]):
    """
    Process options according to question_type and return:
    - options (with original index, text, image)
    - option_mapping (new_index -> old_index)
    """
    option_mapping = {}
    
    if q_type == "yes_or_no":
        options = [{"orig_idx": 0, "text": "yes"},
                   {"orig_idx": 1, "text": "no"},
                   {"orig_idx": 2, "text": "unknown"}]
        option_mapping = [0,1,2]

        return options, option_mapping
    elif q_type == "multiple_choice":
        options = []
        n = max(len(option_texts or []), len(option_images or []))
        for idx in range(n):
            ot = option_texts[idx] if option_texts and idx < len(option_texts) else None
            oi = option_images[idx] if option_images and idx < len(option_images) else None
            options.append({"orig_idx": idx, "text": ot, "image": oi})
        for new_idx, opt in enumerate(options):
            option_mapping[new_idx] = opt["orig_idx"]
        return options, option_mapping


# ===================== general builder function =====================
def _generic_builder(question_json: Dict[str, Any]) -> (List[Dict], Dict[int, int]):
    q_text = question_json["question_text"]
    q_images = question_json.get("question_image_dir_list", [])
    if isinstance(q_images,str):
        q_images = [q_images]
    option_texts = question_json.get("option_text", [])
    option_images = question_json.get("option_image_dir_list", [])
    q_type = question_json["question_type"]
    sub_type = question_json["knowledge"]["knowledge_sub_type"]

    options, option_mapping  = _process_options(question_json['question_id'],q_type, option_texts, option_images)


    return options, option_mapping, q_text,q_images, q_type,sub_type



def build_interface_perception(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt):
    system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given a screenshot, a question, and corresponding options. You need to choose one option as your answer.\n'
        
    messages = [
            {"role":"system",
            "content":[{
                "type":"text","text":system_prompt}]},
            {"role":"user",
            "content":[]
            }
            ]
    messages_debug = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    if q_type == 'yes_or_no': 
        
        
        if q_images:
            for q_image in q_images:
                messages[1]['content'].append(encode_image(q_image))
                messages_debug[1]['content'].append({"type": "image", "image_url": q_image})
        messages[1]['content'].append({ "type":"text","text":q_text+' \n'})
        messages_debug[1]['content'].append({ "type":"text","text":q_text+' \n'})     
        # specific requirement for the answer
       
        if enable_thinking_prompt:
            prompt_text = 'Think step by step. You must respond strictly in JSON format following this schema:{   "thought": "<your reasoning>", "answer": "<yes/no/unknown>" } '
        else:
            prompt_text ='You must respond strictly in JSON format following this schema: {"answer": "<yes/no/unknown>" }'
        messages[1]['content'].append({"type":"text","text":prompt_text})
        messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
        
    if q_type == 'multiple_choice':
        
        
        if q_images:
            for q_image in q_images:
                messages[1]['content'].append(encode_image(q_image))
                messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
        messages[1]['content'].append({ "type":"text","text":q_text+'\n'})
        messages_debug[1]['content'].append({ "type":"text","text":q_text+'\n'})
        for i, opt in enumerate(options):
            label_chr = chr(65 + i)
            # Text line: even if no text, still output the label so the option remains selectable
            text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
            messages[1]['content'].append({ "type":"text","text":text_line+'\n'})
            messages_debug[1]['content'].append({ "type":"text","text":text_line+'\n'})
            
            # Followed by image if present
            if opt.get("image"):
                messages[1]['content'].append(encode_image(opt['image']))
                messages_debug[1]['content'].append({"type": "image", "image_url": opt['image']}) 
        
        prompt_text = 'Which of the above options are correct according to the screenshot?'
        if enable_thinking_prompt:
            prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
        else:
            prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: Answer: {"answer": "<A/B/C/D>" } '
        messages[1]['content'].append({"type":"text","text":prompt_text})
        messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
    return messages,messages_debug
    


def build_interface_perception_ablation(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt,ablation_options,knowledge,annotaiton,q_id,q_visual_text):
    system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given a screenshot, a question, and corresponding options. You need to choose one option as your answer.\n'
        
    messages = [
            {"role":"system",
            "content":[{
                "type":"text","text":system_prompt}]},
            {"role":"user",
            "content":[]
            }
            ]
    messages_debug = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    
    image_path = q_images[0] if isinstance(q_images, list) else q_images
    true_image_path = find_image_path(image_path)    
    draw_annotation(true_image_path,annotaiton,f'AnnotateImage/{q_id}.png')
    ### create bounding box for image_path and save it somewhere
    image_path_with_visual_prompt = f'AnnotateImage/{q_id}.png'
    
    if q_type == 'yes_or_no': 
        
        if ablation_options['visual_prompt'] == False:
            if q_images:
                for q_image in q_images:
                    messages[1]['content'].append(encode_image(q_image))
                    messages_debug[1]['content'].append({"type": "image", "image_url": q_image})
            messages[1]['content'].append({ "type":"text","text":q_text+' \n'})
            messages_debug[1]['content'].append({ "type":"text","text":q_text+' \n'})
        else:
            messages[1]['content'].append(encode_image(image_path_with_visual_prompt))
            messages_debug[1]['content'].append({"type": "image", "image_url": image_path_with_visual_prompt})
            messages[1]['content'].append({ "type":"text","text":q_visual_text+' \n'})
            messages_debug[1]['content'].append({ "type":"text","text":q_visual_text+' \n'})
           
        
        
       
        if enable_thinking_prompt:
            prompt_text = 'Think step by step. You must respond strictly in JSON format following this schema:{   "thought": "<your reasoning>", "answer": "<yes/no/unknown>" } '
        else:
            prompt_text ='You must respond strictly in JSON format following this schema: {"answer": "<yes/no/unknown>" }'
        if ablation_options['knowledge_prompt'] == True:
            prompt_text = prompt_text +f'Tips for answering this question: {knowledge}'
        messages[1]['content'].append({"type":"text","text":prompt_text})
        messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
        
    if q_type == 'multiple_choice':
        
        
        if ablation_options['visual_prompt'] == False:
            if q_images:
                for q_image in q_images:
                    messages[1]['content'].append(encode_image(q_image))
                    messages_debug[1]['content'].append({"type": "image", "image_url": q_image})
            messages[1]['content'].append({ "type":"text","text":q_text+' \n'})
            messages_debug[1]['content'].append({ "type":"text","text":q_text+' \n'})
        else:
            messages[1]['content'].append(encode_image(image_path_with_visual_prompt))
            messages_debug[1]['content'].append({"type": "image", "image_url": image_path_with_visual_prompt})
            messages[1]['content'].append({ "type":"text","text":q_visual_text+' \n'})
            messages_debug[1]['content'].append({ "type":"text","text":q_visual_text+' \n'})
        
        for i, opt in enumerate(options):
            label_chr = chr(65 + i)
            # Text line: even if no text, still output the label so the option remains selectable
            text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
            messages[1]['content'].append({ "type":"text","text":text_line+'\n'})
            messages_debug[1]['content'].append({ "type":"text","text":text_line+'\n'})
            
            # Followed by image if present
            if opt.get("image"):
                messages[1]['content'].append(encode_image(opt['image']))
                messages_debug[1]['content'].append({"type": "image", "image_url": opt['image']}) 
        
        prompt_text = 'Which of the above options are correct according to the screenshot?'
        if enable_thinking_prompt:
            prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
        else:
            prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: Answer: {"answer": "<A/B/C/D>" } '
        if ablation_options['knowledge_prompt'] == True:
            prompt_text = prompt_text +f'Tips for answering this question: {knowledge}'
        messages[1]['content'].append({"type":"text","text":prompt_text})
        messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
    return messages,messages_debug
    
        
        

def build_instruction_understanding(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt):
    
    system_prompt = 'You are a helpful agent.'
    if sub_type == 'GoalInterpretation':
        system_prompt = 'You are a Graphical User Interface (GUI) agent.  You will be given a sequence of screenshots, a task instruction, and three possible answer options: yes, no, unknown. Your goal is to select the best option that indicates whether the task is completed: yes — The task is clearly completed.  no — The task is not completed.  unknown — The screenshots do not provide enough evidence to determine completion.\n'
    elif sub_type == 'TaskPlanning':
        if q_type == 'yes_or_no':
            system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given a screenshot, a question, and corresponding options. You need to choose one option as your answer.\n'
        if q_type == 'multiple_choice':
            system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given a task instruction, a screenshot, several GUI operations, and four options. Your goal is to select the best option that could solve the task.\n'
        
    else:
        system_prompt = 'You are a helpful agent.'
    
    messages = [
            {"role":"system",
            "content":[{
                "type":"text","text":system_prompt}]},
            {"role":"user",
            "content":[]
            }
            ]
    messages_debug = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    
    if sub_type == 'GoalInterpretation':
        if q_type == 'yes_or_no': 
            
            
            for q_image in q_images:
                messages[1]['content'].append(encode_image(q_image))
                messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
            messages[1]['content'].append({ "type":"text","text":f"According to the screenshots above, has the task \"{q_text}\" been completed?"})
            messages_debug[1]['content'].append({ "type":"text","text":f"According to the screenshots above, has the task \"{q_text}\" been completed?"})
            # specific requirement for the answer
            prompt_text = ''
            if enable_thinking_prompt:
                prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {   "thought": "<your reasoning>", "answer": "<yes/no/unknown>" } '
            else:
                prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: Answer: Answer: {"answer": "<yes/no/unknown>" } '
            messages[1]['content'].append({"type":"text","text":prompt_text})
            messages_debug[1]['content'].append({"type":"text","text":prompt_text})
            
    
    if sub_type == 'TaskPlanning':
        if q_type == 'multiple_choice': 
            
            for q_image in q_images:
                messages[1]['content'].append(encode_image(q_image))
                messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
            messages[1]['content'].append({ "type":"text","text":f"{q_text}"})
            messages_debug[1]['content'].append({ "type":"text","text":f"{q_text}"})
            for i, opt in enumerate(options):
                label_chr = chr(65 + i)
                # Text line: even if no text, still output the label so the option remains selectable
                text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                messages[1]['content'].append({ "type":"text","text":text_line + '\n'})
                messages_debug[1]['content'].append({ "type":"text","text":text_line+ '\n'})
            
            
            prompt_text = 'Which of the above options are correct according to the screenshots?'
            if enable_thinking_prompt:
                prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
            else:
                prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: Answer: {"answer": "<A/B/C/D>" } '
            messages[1]['content'].append({"type":"text","text":prompt_text})
            messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
        if q_type == 'yes_or_no': 
            
            for q_image in q_images:
                messages[1]['content'].append(encode_image(q_image))
                messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
            messages[1]['content'].append({ "type":"text","text":f"{q_text}"})
            messages_debug[1]['content'].append({ "type":"text","text":f"{q_text}"})
            prompt_text = ''
            if enable_thinking_prompt:
                prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {   "thought": "<your reasoning>", "answer": "<yes/no/unknown>" } '
            else:
                prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: Answer: Answer: {"answer": "<yes/no/unknown>" } '
            messages[1]['content'].append({"type":"text","text":prompt_text})
            messages_debug[1]['content'].append({"type":"text","text":prompt_text})
            
        
    return messages,messages_debug
    


def build_interaction_prediction(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt,os_type,question_id):
    if os_type == 'Android':
        action_space_defination = '''
        ## Action Space\n
        - click(point='x1 y1')
        - long_press(point='x1 y1')
        - type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n
        '''
    else: 
        action_space_defination = '''
        Action Space\n
        - click(point='x1 y1'): left click a position on the screen. 
        - left_double(point='x1 y1'): left double click a position on the screen. 
        - right_single(point='x1 y1'): right single click a position on the screen. 
        - drag(start_point='x1 y1', end_point='x2 y2'): drag the mouse from one position to another. 
        - hotkey(key='ctrl c'): keyboard shortcut, split keys with spaces
        - type(content='xxx'): type an answer, use escape characters (\', \", \\n) when needed. Add \\n at the end if it is the final submission.
        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n
        '''
    
    image_info_string = get_image_size(q_images[0])
    system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given screenshot of an application, a question, and corresponding options. You need to choose one option as your answer for the question based on your knowledge of GUI interaction.\n'
    if sub_type == 'ActionEffect':
        system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given a screenshot, action descriptions, and multiple options, each containing an image. After performing one action on the screenshot, your goal is to select the option that correctly corresponds to the resulting screenshot after performing the action. Below is a short description of the action space:\n\n' + action_space_defination + image_info_string
    if sub_type == 'ActionPrediction':
        system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given two consecutive screenshots of the GUI, action descriptions, and multiple options. Your goal is to select which action was performed to transition from the first screenshot to the second. If the description specifies an action type, select the correct parameter value for the given action.\n\n' + action_space_defination+ image_info_string
        
    messages = [
            {"role":"system",
            "content":[{
                "type":"text","text":system_prompt}]},
            {"role":"user",
            "content":[]
            }
            ]
    messages_debug = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    if sub_type == 'ActionEffect':
          
        if q_images:
            for q_image in q_images:
                q_image = find_image_path(q_image) # resolve actual path for this image
                image = _ensure_image_exists(q_image)
                action_type = q_text.replace('ActionEffect:','')
                image = draw_action(image,action_type)
                temp_save_path = f"AnnotateImage/{question_id}.png"
                cv2.imwrite(temp_save_path, image)  # type: ignore[attr-defined]  # pylint: disable=no-member
                messages[1]['content'].append(encode_image(temp_save_path))
                messages_debug[1]['content'].append({"type": "image", "image_url": temp_save_path})  # updated
        
        messages[1]['content'].append({ "type":"text","text":'Above is the current screenshot.\n' })
        messages_debug[1]['content'].append({ "type":"text","text":'Above is the current screenshot.\n'}) 
                
        action_type = q_text.replace('ActionEffect: ','')
        indication_text = ''
        if action_type.startswith('click') or action_type.startswith('left_double') or action_type.startswith('right_single') or action_type.startswith('scroll') or action_type.startswith('drag') or action_type.startswith('long_press'):
            indication_text = '(as drawn in the initial screenshot.)'
        messages[1]['content'].append({ "type":"text","text":f'After I perform the described action \'{action_type}\' {indication_text}, which of the following options correctly corresponds to the resulting screenshot?\n' })
        messages_debug[1]['content'].append({ "type":"text","text":f'After I perform the described action \'{action_type}\'{indication_text}, which of the following options correctly corresponds to the resulting screenshot?\n' })
        
        for i, opt in enumerate(options):
            label_chr = chr(65 + i)
            # Text line: even if no text, still output the label so the option remains selectable
            text_line = f"{label_chr}."
            messages[1]['content'].append({ "type":"text","text":text_line + ' \n'})
            messages_debug[1]['content'].append({ "type":"text","text":text_line+ ' \n'})
            
            # Followed by image if present
            if opt.get("image"):
                messages[1]['content'].append(encode_image(opt['image']))
                messages_debug[1]['content'].append({"type": "image", "image_url": opt['image']})  # updated

        if enable_thinking_prompt:
            prompt_text = 'Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
        else:
            prompt_text =  'You must respond strictly in JSON format following this schema: Answer: {"answer": "<A/B/C/D>" } '
        messages[1]['content'].append({"type":"text","text":prompt_text})
        messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
    
    
    if sub_type == 'ActionPrediction':
        
        if q_type == 'multiple_choice':
            prompt_text = ''
            if 'ActionPrediction-Parameter:' in q_text:  
                action_type = q_text.replace('ActionPrediction-Parameter: ','')     
                if action_type.startswith('click') or action_type.startswith('left_double') or action_type.startswith('right_single') or action_type.startswith('scroll') or action_type.startswith('drag') or action_type.startswith('long_press'): 
                    action_type = q_text.replace('ActionPrediction-Parameter: ','')
                    prompt_text = f'Above are two consecutive screenshots, Your task is to select the option containing the right parameter value of the given action \'{action_type}\' to transition from the first to the second screenshot.'
                    # print(action_type)

                    q_image_2 =   q_images[1]
                    q_images_1 = q_images[0]
                    q_images_temp = find_image_path(q_images_1)
                    q_images_temp = _ensure_image_exists(q_images_temp)
                    for i, opt in enumerate(options):
                        # text message 
                        label_chr = chr(65 + i)
                        text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                        option_text = opt['text']
                        action_description = f'{action_type}({option_text})'
                        q_images_temp = draw_action(q_images_temp, action_description, str(label_chr))
                        
                    temp_save_path  = f"AnnotateImage/{question_id}.png"
                    cv2.imwrite(temp_save_path, q_images_temp)  # type: ignore[attr-defined]  # pylint: disable=no-member
                    messages[1]['content'].append(encode_image(temp_save_path))
                    messages_debug[1]['content'].append({"type": "image", "image_url": temp_save_path})

                    messages[1]['content'].append(encode_image(q_image_2))
                    messages_debug[1]['content'].append({"type": "image", "image_url": q_image_2})

                    
                    messages[1]['content'].append({ "type":"text","text":prompt_text})
                    messages_debug[1]['content'].append({ "type":"text","text":prompt_text})
                
                    for i, opt in enumerate(options):
                        # text message 
                        label_chr = chr(65 + i)
                        text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                        messages[1]['content'].append({ "type":"text","text":text_line + ' \n' + 'As is drawn in the first screenshot.'})
                        messages_debug[1]['content'].append({ "type":"text","text":text_line + ' \n'+ 'As is drawn in the first screenshot.'})
                else: 
                    
                    if q_images:
                        for q_image in q_images:
                            messages[1]['content'].append(encode_image(q_image))
                            messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
                    action_type = q_text.replace('ActionPrediction-Parameter: ','')
                    prompt_text = f'Above are two consecutive screenshots, Your task is to select the parameter value of the given action \'{action_type}\' to transition from the first to the second screenshot.'
                    # print(action_type)
                    
                    messages[1]['content'].append({ "type":"text","text":prompt_text})
                    messages_debug[1]['content'].append({ "type":"text","text":prompt_text})
                    for i, opt in enumerate(options):
                        label_chr = chr(65 + i)
                        # Text line: even if no text, still output the label so the option remains selectable
                        text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                        messages[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                        messages_debug[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                        
                        # # Followed by image if present
                        # if opt.get("image"):
                        #     messages[1]['content'].append(encode_image(opt['image']))
                        #     messages_debug[1]['content'].append({"type": "image", "image_url": opt['image']})  # updated
                
                        
            if 'ActionPrediction-Type:' in q_text:
                prompt_text = 'Above are two consecutive screenshots, Your task is to select which action is performed in order to transition from the first screenshot to the second.'
                
                if q_images:
                    for q_image in q_images:
                        messages[1]['content'].append(encode_image(q_image))
                        messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
                messages[1]['content'].append({ "type":"text","text":prompt_text})
                messages_debug[1]['content'].append({ "type":"text","text":prompt_text})
                
                for i, opt in enumerate(options):
                    label_chr = chr(65 + i)
                    # Text line: even if no text, still output the label so the option remains selectable
                    text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                    messages[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                    messages_debug[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                    
            prompt_text = 'Which of the above options are correct according to the screenshots?'
            if enable_thinking_prompt:
                if len(options) == 7:
                    prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D/E/F/G>" } '
                else:
                    prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
            else:
                if len(options) == 7:
                    prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D/E/F/G>" } '
                else:
                    prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
        
            messages[1]['content'].append({"type":"text","text":prompt_text})
            messages_debug[1]['content'].append({"type":"text","text":prompt_text})
            
    return messages,messages_debug
    


def build_interaction_prediction_no_visual_prompt(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt,os_type,question_id):
    if os_type == 'Android' or os_type == 'IOS':
        action_space_defination = '''
        ## Action Space\n
        - click(point='x1 y1')
        - long_press(point='x1 y1')
        - type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n
        '''
    else: 
        action_space_defination = '''
        Action Space\n
        - click(point='x1 y1'): left click a position on the screen. 
        - left_double(point='x1 y1'): left double click a position on the screen. 
        - right_single(point='x1 y1'): right single click a position on the screen. 
        - drag(start_point='x1 y1', end_point='x2 y2'): drag the mouse from one position to another. 
        - hotkey(key='ctrl c'): keyboard shortcut, split keys with spaces
        - type(content='xxx'): type an answer, use escape characters (\', \", \\n) when needed. Add \\n at the end if it is the final submission.
        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n
        '''
    
    image_info_string = get_image_size(q_images[0])
    system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given screenshot of an application, a question, and corresponding options. You need to choose one option as your answer for the question based on your knowledge of GUI interaction.\n'
    if sub_type == 'ActionEffect':
        system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given a screenshot, action descriptions, and multiple options, each containing an image. After performing one action on the screenshot, your goal is to select the option that correctly corresponds to the resulting screenshot after performing the action. Below is a short description of the action space:\n\n' + action_space_defination + image_info_string
    if sub_type == 'ActionPrediction':
        system_prompt = 'You are a Graphical User Interface (GUI) agent. You will be given two consecutive screenshots of the GUI, action descriptions, and multiple options. Your goal is to select which action was performed to transition from the first screenshot to the second. If the description specifies an action type, select the correct parameter value for the given action.\n\n' + action_space_defination+ image_info_string
        
    messages = [
            {"role":"system",
            "content":[{
                "type":"text","text":system_prompt}]},
            {"role":"user",
            "content":[]
            }
            ]
    messages_debug = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    if sub_type == 'ActionEffect':
        
        if q_images:
            for q_image in q_images:
                q_image = find_image_path(q_image) # resolve actual path for this image
                # image = _ensure_image_exists(q_image)
                # action_type = q_text.replace('ActionEffect:','')
                # image = draw_action(image,action_type)
                messages[1]['content'].append(encode_image(q_image))
                messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  # updated
        if q_type == 'multiple_choice':
            messages[1]['content'].append({ "type":"text","text":'Above is a current screenshot.\n' })
            messages_debug[1]['content'].append({ "type":"text","text":'Above is a current screenshot.\n'})      
        action_type = q_text.replace('ActionEffect: ','')
        indication_text = ''
        # if action_type.startswith('click') or action_type.startswith('left_double') or action_type.startswith('right_single') or action_type.startswith('scroll') or action_type.startswith('drag') or action_type.startswith('long_press'):
        #     indication_text = '(as drawn in the initial screenshot.)'
        messages[1]['content'].append({ "type":"text","text":f'After I perform the described action \'{action_type}\' {indication_text}, which of the following options correctly corresponds to the resulting screenshot?\n' })
        messages_debug[1]['content'].append({ "type":"text","text":f'After I perform the described action \'{action_type}\'{indication_text}, which of the following options correctly corresponds to the resulting screenshot?\n' })
        
        for i, opt in enumerate(options):
            label_chr = chr(65 + i)
            # Text line: even if no text, still output the label so the option remains selectable
            text_line = f"{label_chr}."
            messages[1]['content'].append({ "type":"text","text":text_line + ' \n'})
            messages_debug[1]['content'].append({ "type":"text","text":text_line+ ' \n'})
            
            # Followed by image if present
            if opt.get("image"):
                messages[1]['content'].append(encode_image(opt['image']))
                messages_debug[1]['content'].append({"type": "image", "image_url": opt['image']})  # updated

        if enable_thinking_prompt:
            prompt_text = 'Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
        else:
            prompt_text =  'You must respond strictly in JSON format following this schema: Answer: {"answer": "<A/B/C/D>" } '
        messages[1]['content'].append({"type":"text","text":prompt_text})
        messages_debug[1]['content'].append({"type":"text","text":prompt_text})
        
    
    
    if sub_type == 'ActionPrediction':
    
        
        if q_type == 'multiple_choice':
            prompt_text = ''
            if 'ActionPrediction-Parameter:' in q_text:
                if q_images:
                    for q_image in q_images:
                        messages[1]['content'].append(encode_image(q_image))
                        messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
                        
                action_type = q_text.replace('ActionPrediction-Parameter: ','')
                prompt_text = f'Above are two consecutive screenshots, Your task is to select the parameter value of the given action \'{action_type}\''
                messages[1]['content'].append({ "type":"text","text":prompt_text})
                messages_debug[1]['content'].append({ "type":"text","text":prompt_text})
                    
                for i, opt in enumerate(options):
                    label_chr = chr(65 + i)
                    # Text line: even if no text, still output the label so the option remains selectable
                    text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                    messages[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                    messages_debug[1]['content'].append({ "type":"text","text":text_line + ' \n'})
            
            if 'ActionPrediction-Type:' in q_text:
                # return {}, {}
                prompt_text = 'Above are two consecutive screenshots, Your task is to select which action is performed in order to transition from the first screenshot to the second.'
                
                if q_images:
                    for q_image in q_images:
                        messages[1]['content'].append(encode_image(q_image))
                        messages_debug[1]['content'].append({"type": "image", "image_url": q_image})  
                messages[1]['content'].append({ "type":"text","text":prompt_text})
                messages_debug[1]['content'].append({ "type":"text","text":prompt_text})
                
                for i, opt in enumerate(options):
                    label_chr = chr(65 + i)
                    # Text line: even if no text, still output the label so the option remains selectable
                    text_line = f"{label_chr}. {opt['text']}" if opt.get("text") else f"{label_chr}."
                    messages[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                    messages_debug[1]['content'].append({ "type":"text","text":text_line + ' \n'})
                    
            prompt_text = 'Which of the above options are correct according to the screenshots?'
            if enable_thinking_prompt:
                if len(options) == 7:
                    prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D/E/F/G>" } '
                else:
                    prompt_text = prompt_text + ' Think step by step. You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
            else:
                if len(options) == 7:
                    prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D/E/F/G>" } '
                else:
                    prompt_text = prompt_text + ' You must respond strictly in JSON format following this schema: {"thought": "<your reasoning>", "answer": "<A/B/C/D>" } '
        
            messages[1]['content'].append({"type":"text","text":prompt_text})
            messages_debug[1]['content'].append({"type":"text","text":prompt_text})
                
            
    return messages,messages_debug
    

   
    

# ===================== Main entry =====================
def generate_message(question_json: Dict[str, Any],enable_thinking_prompt,ablation_options) -> Dict[str, Any]:
    knowledge_type = question_json["knowledge"]["knowledge_type"]
    options, option_mapping, q_text,q_images, q_type,sub_type = _generic_builder(question_json)
    # options include option texts and option images,
    # options have been shuffled according to the option mapping. 
    # yes or no question has no shuffled options. 
    if knowledge_type == 'InterfacePerception':
        knowledge = question_json['needed_knowledge']
        annotation = question_json['annotation']
        if ablation_options['visual_prompt'] == False and ablation_options['knowledge_prompt'] == False:
            messages,message_debug = build_interface_perception(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt)
        else:
            messages,message_debug = build_interface_perception_ablation(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt,ablation_options,knowledge,annotation,question_json['question_id'],question_json['augmented_question'])
        return messages,option_mapping,message_debug
    if knowledge_type == 'InstructionUnderstanding':
        messages,message_debug = build_instruction_understanding(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt)
        return messages,option_mapping,message_debug
    if knowledge_type == 'InteractionPrediction':
        visual_prompt = ablation_options['visual_prompt']
        if visual_prompt:
            messages,message_debug = build_interaction_prediction(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt,question_json['os_type'],question_json['question_id'])
        else:
            messages,message_debug = build_interaction_prediction_no_visual_prompt(options, option_mapping, q_text,q_images, q_type,sub_type,enable_thinking_prompt,question_json['os_type'],question_json['question_id'])
        return messages,option_mapping,message_debug
    else:
        print("knowledge_type not recognized",knowledge_type)
        exit()
        return None,None,None
