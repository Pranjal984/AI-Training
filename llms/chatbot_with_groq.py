import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

if __name__ == '__main__':
    api_key = os.getenv('GROQ_API_KEY')
    if api_key is None:
        print('You need to set your GROQ_API_KEY environment variable')
        exit(1)

    model = "llama3-70b-8192"
    client = Groq(api_key=api_key)

    user_input = input("You: ")

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        temperature=0.7
    )

    print("\n--- RESPONSE ---")
    if chat_response and chat_response.choices:
        print("AI:", chat_response.choices[0].message.content)
    else:
        print(" No response or choices from the API. Response was:", chat_response)
