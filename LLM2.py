import openai
import os

API_KEY = os.getenv("API_KEY")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.intelligence.io.solutions/api/v1/",
)
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": "Ты - Джарвис, как из фильма 'Железный человек'. но я привык назвать тебя 'кен', так что не заостряй внимание на этом. Кстати, когда отвечаешь не забудь разбавлять ответ шуткой или сарказмом. и ещё, если твой ответ включает в себя код, не пиши его. говори всё на словах"},
            {"role": "user", "content": f"Привет, {prompt}"},
        ],
        temperature=0.7,
        stream=False,
        max_completion_tokens=350
    )

    return response.choices[0].message.content


# print(generate_answer("Привет. расскажи про квантовую механику"))