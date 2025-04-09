import os
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()
if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    user_input = input("You: ")
    stream_response = client.chat.stream(
        model=model,
        messages=[
            # {
            #     "role": "system",
            #     "content" : "My name is Pranjal Tripathi"
            # },
            # {
            #     "role": "user",
            #     "content": "What is the capital of Madhya Pradesh?",
            # },
            {
                "role": "user",
                "content": user_input,
            }

        ]
    )
    for chunk in stream_response:
        # print(chunk.data.choices[0].delta.content)
        print(chunk.data.choices[0].delta.content, end="")