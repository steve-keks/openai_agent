from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import dotenv
import os
import openai
from litellm import completion


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Dict] = None
    finish_reason: str

class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[Choice]
    service_tier: str
    usage: Usage

# Example usage
data = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?",
        },
        "logprobs": None,
        "finish_reason": "stop"
    }],
    "service_tier": "default",
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21,
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    }
}



dotenv.load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("API_KEY"),
)

'''
chat_completion = client.chat.completions.create(
    max_completion_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Finland?,"
        }
    ],
    model="gpt-4o",
)


print(type(chat_completion))

# convert the response to a dictionary
chat_completion_dict = chat_completion.to_dict()
'''


litellm_resp = completion(
    api_key=os.getenv("api_key"),
    model="openai/gpt-4",
    messages=[{ "content": "What is the capital of Finland?","role": "user"}],
    max_tokens=100
)

# convert litellm response to a dictionary
litellm_dict = litellm_resp.to_dict()




try:
    #chat_completion = ChatCompletion(**chat_completion_dict)
    #print("The openai data is valid.")
    chat_completion = ChatCompletion(**litellm_dict)
    print("The litellm data is valid.")
    print(litellm_resp)

except ValueError as e:
    print("The data is not valid.")
    raise ValueError(e)