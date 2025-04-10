import os
from mistralai import Mistral
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("Loaded MONGO_URI:", os.getenv("MONGO_URI"))


class ChatBot:
    def __init__(self, api_key, model, profile_slug="slug"):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.mistral_client = Mistral(api_key=api_key)
        self.initialize_context()

    def initialize_context(self):
        mongo_url = os.getenv("MONGO_URI")
        client = MongoClient(mongo_url)
        db = client["app-dev"]
        collection = db["profiles"]
        profiles = collection.find()

        profile_strings = []
        for profile in profiles:
            profile_str = f"Name: {profile.get('firstName', '')} {profile.get('lastName', '')}\n" \
                          f"Expertise: {profile.get('areaOfExpertise', '')}\n" \
                          f"Location: {profile.get('currentLocation', '')}\n" \
                          f"Member Since: {profile.get('businessMemberSince', '')}\n" \
                          f"Summary: {profile.get('carrierSummary', '')}\n"
            profile_strings.append(profile_str)

        full_context = "\n\n".join(profile_strings)
        self.conversation_history.append({
            "role": "system",
            "content": f"The following are the details of professionals in the system:\n\n{full_context}"
        })

    def Get_user_input(self):

        user_input = input("\n")
        user_message = {
            "role": "user",
            "content": user_input
        }
        self.conversation_history.append(user_message)
        return user_message

    def send_request(self):
        stream_response = self.mistral_client.chat.stream(
            model=self.model,
            messages=self.conversation_history
        )
        buffer = ""
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content
            if content:
                buffer += content
                print(content, end="")

    def run(self):
        while True:
            self.Get_user_input()
            self.send_request()


if __name__ == "__main__":
    chat_bot = ChatBot(os.getenv("MISTRAL_API_KEY"), model="mistral-large-latest")
    chat_bot.initialize_context()
    chat_bot.run()
