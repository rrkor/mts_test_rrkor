from settings import API_KEY, MODELS, TEMPERATURE, MAX_TOKENS, SYSTEM_MESSAGE, TOP_P, USER_MESSAGE, REFERENCE_ANSWER
from pricing import PRICING
from openai import OpenAI
import json

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

def calc_bleu(hyp, ref):
    from sacrebleu.metrics import BLEU
    return BLEU(effective_order=True).sentence_score(hyp, [ref]).score / 100

def calc_rouge(hyp, ref):
    from rouge import Rouge
    score = Rouge().get_scores(hyp, ref)[0]
    return {
        "rouge_1_f": score["rouge-1"]["f"],
        "rouge_2_f": score["rouge-2"]["f"],
        "rouge_l_f": score["rouge-l"]["f"],
    }



def chat_with_gpt(api_key, user_message, model, temperature, max_tokens, system_message, top_p, reference_summary):
    client = OpenAI(api_key=api_key)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens
    )
    assistant_message = response.choices[0].message.content
    usage = response.usage
    cost = calculate_cost(usage.prompt_tokens, usage.completion_tokens, model)
    # Метрики
    bleu = calc_bleu(assistant_message, reference_summary)
    rouge = calc_rouge(assistant_message, reference_summary)
    return {
        "response": assistant_message,
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "total_cost": cost["total_cost"],
        "bleu": bleu,
        "rouge_1_f": rouge["rouge_1_f"],
        "rouge_2_f": rouge["rouge_2_f"],
        "rouge_l_f": rouge["rouge_l_f"],
        "error": ""
    }

results = []
for model in MODELS:
    try:
        result = chat_with_gpt(
            API_KEY, USER_MESSAGE, model,
            TEMPERATURE, MAX_TOKENS, SYSTEM_MESSAGE, TOP_P, REFERENCE_ANSWER
        )
    except Exception as e:
        result = {
            "response": "",
            "model": model,
            "input_tokens": "-",
            "output_tokens": "-",
            "total_tokens": "-",
            "total_cost": "-",
            "bleu": None,
            "rouge_1_f": None,
            "rouge_2_f": None,
            "rouge_l_f": None,
            "error": str(e)
        }
    results.append(result)

with open("results_metrics.json", "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

for r in results:
    print(f"\nМодель: {r['model']}")
    if r["error"]:
        print(f"Ошибка: {r['error']}")
        continue
    print(f"Входные токены: {r['input_tokens']}")
    print(f"Выходные токены: {r['output_tokens']}")
    print(f"Всего токенов: {r['total_tokens']}")
    print(f"Стоимость запроса: {r['total_cost']} $")
    print("Метрики:")
    print(f"BLEU: {r['bleu']:.3f}")
    print(f"ROUGE-1-F: {r['rouge_1_f']:.3f}")
    print(f"ROUGE-2-F: {r['rouge_2_f']:.3f}")
    print(f"ROUGE-L-F: {r['rouge_l_f']:.3f}")
    print("Ответ:")
    print(r["response"])


def model_best_quality(results):
    best = None
    best_score = -1
    for r in results:
        metrics = [r.get('bleu'), r.get('rouge_1_f'), r.get('rouge_2_f'), r.get('rouge_l_f')]
        if any(m is None for m in metrics): continue
        avg = sum(metrics) / len(metrics)
        if avg > best_score:
            best_score = avg
            best = r
    return best, best_score

def model_cost_benefit_analysis(results):
    best = None
    best_value = -1
    for r in results:
        metrics = [r.get('bleu'), r.get('rouge_1_f'), r.get('rouge_2_f'), r.get('rouge_l_f')]
        if any(m is None for m in metrics): continue
        avg_metric = sum(metrics) / len(metrics)
        cost = r.get('total_cost', None)
        try:
            cost_float = float(cost)
        except:
            continue
        value = avg_metric / cost_float if cost_float > 0 else 0
        if value > best_value:
            best_value = value
            best = r
    return best, best_value





best_by_quality, quality_score = model_best_quality(results)
print(f"""\nЛучшая модель по среднему качеству
Модель: {best_by_quality['model']}
Средний балл: {quality_score:.3f}
BLEU: {best_by_quality['bleu']:.3f}
ROUGE-1-F: {best_by_quality['rouge_1_f']:.3f}
ROUGE-2-F: {best_by_quality['rouge_2_f']:.3f}
ROUGE-L-F: {best_by_quality['rouge_l_f']:.3f}
Стоимость запроса: {best_by_quality['total_cost']} $
Ответ:
{best_by_quality['response']}""")

best_by_value, value_score = model_cost_benefit_analysis(results)
print(f"""\nСамая выгодная модель
Модель: {best_by_value['model']}. метрика/доллар: {value_score:.1f}
BLEU: {best_by_value['bleu']:.3f}
ROUGE-1-F: {best_by_value['rouge_1_f']:.3f}
ROUGE-2-F: {best_by_value['rouge_2_f']:.3f}
ROUGE-L-F: {best_by_value['rouge_l_f']:.3f}
Стоимость запроса: {best_by_value['total_cost']} $
Ответ:
{best_by_value['response']}""")

