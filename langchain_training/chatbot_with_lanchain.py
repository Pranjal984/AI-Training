import os
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_mistralai.chat_models import ChatMistralAI  # Requires `langchain-mistralai`

load_dotenv()

class ProfilesChatBot:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.chat_model = ChatMistralAI(api_key=api_key, model_name=model_name, streaming=True)
        self.system_context = self.load_profiles_context()
        self.prompt = self.build_prompt()

    def load_profiles_context(self):
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

        return "\n\n".join(profile_strings)

    def build_prompt(self):
        return PromptTemplate.from_template(
            """You are a helpful assistant. You have access to professional profiles in the system.

Profiles:
{profiles}

User Question:
{question}

Answer:"""
        )

    def run(self):
        print("Type your question (type 'exit' to quit):\n")
        while True:
            user_input = input("\n You: ")
            if user_input.strip().lower() == "exit":
                break

            chain: Runnable = self.prompt | self.chat_model | StrOutputParser()
            response = ({
                "profiles": self.system_context,
                "question": user_input
            })
            for token in chain.stream(response):
                print(token, end="")


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    model = "mistral-large-latest"
    bot = ProfilesChatBot(api_key, model)
    bot.run()
