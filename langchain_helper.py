from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from secret_key import openai_api_key

import os
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature = 0.7)

def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template = "I want to open an {cuisine} Restaurant. Suggest a fancy name for this."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="""Suggest some menut items for {restaurant_name}. Return it as a comma seperated list"""
    )

    item_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains = [name_chain, item_chain],          #sequence does matter
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items']
    )

    response = chain({'cuisine':cuisine})

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Chinese"))