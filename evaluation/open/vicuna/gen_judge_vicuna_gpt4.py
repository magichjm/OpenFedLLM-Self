import re
import json
import openai
from openai.error import (
    RateLimitError,
    InvalidRequestError,
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    APIError
)
import os
import time

# ============ 错误重试机制 ============

def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1

# ============ 将英文数字转换为数字 ============

def words_to_numbers(word):
    # 将英文数字转换为对应的数字
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10
    }
    return word_map.get(word.lower(), None)  # 如果没有匹配，返回 None

# ============ 提取评分 ============

def get_rating_from_text(response_text):
    """
    提取评分，从 [ ] 包裹的数字 或者 从最后一个单词往前查找
    """
    # 1. 优先提取被 [[ ]] 包裹的数字
    rating_match = re.search(r"\[\[(\d+)\]\]", response_text)
    if rating_match:
        return rating_match.group(1)  # 返回提取的数字

    # 2. 如果没有 [[ ]]，则从文本最后开始向前查找
    words = response_text.split()  # 按空格分割文本
    for word in reversed(words):
        # 检查是否是数字 0-10 或者英文数字 zero-ten
        if word.isdigit():
            num = int(word)
            if 0 <= num <= 10:
                return str(num)
        else:
            # 检查是否是英文数字
            num = words_to_numbers(word)
            if num is not None:
                return str(num)

    return None  # 如果没有找到有效的评分，返回 None

# ============ GPT-4.1 调用函数 ============

def completion(messages, judger):
    success = False
    response = {}
    while not success:
        try:
            response = openai.ChatCompletion.create(
                model=judger,
                messages=messages,
                temperature=0.2,
                max_tokens=2048
            )
            success = True
        except (RateLimitError, Timeout, APIConnectionError, APIError, ServiceUnavailableError) as e:
            print(e)
            retry_time = get_retry_time(str(e))
            time.sleep(retry_time)
        except InvalidRequestError as e:
            print(e)
            success = True
            response = {"choices": []}
        except Exception as e:
            print(e)
            success = True
            response = {"choices": []}
    return response

# ============ 主流程 ============

def main():
    model_answer = "alpaca-gpt4_50000_fedavg_c5s2_i30_b16a1_l512_r8a16_20250416000539_200.json"
    judger = "gpt-4.1"
    response_path = f"model_answer/{model_answer}"
    save_path = f"model_judgment/{judger}_{model_answer}"

    with open(response_path, "r", encoding="utf-8") as f:
        model_outputs = json.load(f)

    template = (
        "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. "
        "Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response "
        "on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"
        "[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
    )

    print(f"judger: {judger}\ntemplate:\n{template}")
    judge_list = []
    count = 0

    for output in model_outputs:
        count += 1
        current_prompt = template.format(
            question=output["instruction"],
            answer=output["output"]
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant, that ranks models by the quality of their answers."},
            {"role": "user", "content": current_prompt}
        ]

        response = completion(messages, judger)
        response_text = response["choices"][0]["message"]["content"] if response.get("choices") else ""

        # 使用 get_rating_from_text 函数提取评分
        rating = get_rating_from_text(response_text)

        record_sample = {
            "question": output["instruction"],
            "output": output["output"],
            "response": response_text,  # 只存储 response
            "rating": rating,  # 使用提取的评分
            "judger": judger
        }

        judge_list.append(record_sample)

        print("=" * 50, count, "=" * 50)
        print(json.dumps(record_sample, indent=2, ensure_ascii=False))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(judge_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
