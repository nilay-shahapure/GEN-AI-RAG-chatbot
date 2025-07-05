# check_key.py
import os, openai
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    models = openai.models.list()         
    print("✅  Key is valid! You have access to:")
    for m in models.data[:5]:              
        print("   •", m.id)
except openai.error.AuthenticationError as e:
    print("❌  Invalid key or wrong org:", e)
except Exception as e:
    print("⚠️  Something else went wrong:", e)


resp = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content":"ping"}],
    max_tokens=1
)
print("✅  Key works, model replied:", resp.choices[0].message.content)

from agno.embedder.openai import OpenAIEmbedder
import  pprint

embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))

try:
    out = embedder.get_embedding(["test string"])
    pprint.pprint(out)           
except Exception as e:
    print("Embed error:", e.__class__.__name__, "-", e)
