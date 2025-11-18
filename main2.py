from settings import API_KEY, MODELS, TEMPERATURE, MAX_TOKENS, SYSTEM_MESSAGE, TOP_P, USER_MESSAGE
from pricing import PRICING
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import textstat

def calculate_cost(input_tokens, output_tokens, model):
    if model not in PRICING:
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    input_cost = (input_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (output_tokens / 1_000_000) * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6)
    }

def chat_with_gpt(api_key, user_message, model, temperature, max_tokens, system_message, top_p):
    client = OpenAI(api_key=api_key)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    assistant_message = response.choices[0].message.content
    usage = response.usage
    cost = calculate_cost(usage.prompt_tokens, usage.completion_tokens, model)
    return {
        "response": assistant_message,
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "total_cost": cost["total_cost"],
        "error": ""
    }

def evaluate_semantic_similarity(original_text, generated_text):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    emb_orig = model.encode(original_text)
    emb_gen = model.encode(generated_text)
    cosine_sim = util.pytorch_cos_sim(emb_orig, emb_gen)
    return float(cosine_sim)

def evaluate_readability(generated_text):
    textstat.set_lang("ru_RU")
    score = textstat.flesch_kincaid_grade(generated_text)
    return score






results = []
for model in MODELS:
    try:
        result = chat_with_gpt(
            API_KEY, USER_MESSAGE, model,
            TEMPERATURE, MAX_TOKENS, SYSTEM_MESSAGE, TOP_P
        )
        result["semantic_similarity"] = evaluate_semantic_similarity(USER_MESSAGE, result["response"])
        result["readability_score"] = evaluate_readability(result["response"])
    except Exception as e:
        result = {
            "response": "",
            "model": model,
            "input_tokens": "-",
            "output_tokens": "-",
            "total_tokens": "-",
            "total_cost": "-",
            "semantic_similarity": None,
            "readability_score": None,
            "error": str(e)
        }
    results.append(result)

for r in results:
    print(f"\nМодель: {r['model']}")
    if r["error"]:
        print(f"Ошибка: {r['error']}")
        continue
    print(f"Входные токены: {r['input_tokens']}")
    print(f"Выходные токены: {r['output_tokens']}")
    print(f"Всего токенов: {r['total_tokens']}")
    print(f"Стоимость запроса: {r['total_cost']} $")
    print(f"Семантическая близость (0-1): {r['semantic_similarity']:.3f}")
    print(f"Индекс читаемости Флеша-Кинкейда: {r['readability_score']:.2f}")
    print("Ответ:")
    print(r["response"])

def model_best_similarity(results):
    best = None
    best_score = -1
    for r in results:
        sim = r.get('semantic_similarity')
        if sim is not None and sim > best_score:
            best_score = sim
            best = r
    return best, best_score

def model_best_readable(results):
    best = None
    best_score = float('inf')
    for r in results:
        read = r.get('readability_score')
        if read is not None and read < best_score:
            best_score = read
            best = r
    return best, best_score

best_by_similarity, similarity_score = model_best_similarity(results)
print(f"""Лучшая модель по сохранению смысла:
Модель: {best_by_similarity['model']}
Семантическая близость: {similarity_score:.3f}
Стоимость: {best_by_similarity['total_cost']} $
Ответ:
{best_by_similarity['response']}""")

best_by_readability, readability_score = model_best_readable(results)
print(f"""Самая понятная модель:
Модель: {best_by_readability['model']}
Индекс читаемости: {readability_score:.2f}
Стоимость: {best_by_readability['total_cost']} $
Ответ:
{best_by_readability['response']}""")
