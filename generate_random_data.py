import csv, random, os

os.makedirs("data", exist_ok=True)
path = "data/news_with_authors.csv"

topics = [
    "економіка України", "політична реформа", "розвиток штучного інтелекту",
    "нові освітні програми", "енергетична безпека", "сучасна медицина"
]

fake_patterns = [
    "Анонімні джерела повідомляють, що {topic} знаходиться під загрозою.",
    "У соцмережах поширюється чутка про {topic}.",
    "Користувачі мережі стверджують, що уряд приховує дані про {topic}.",
    "Відео у TikTok показує, що {topic} призведе до катастрофи."
]

real_patterns = [
    "Міністерство оприлюднило звіт про стан {topic}.",
    "Експерти заявили, що {topic} розвивається стабільно.",
    "Офіційні джерела підтвердили позитивні зміни у сфері {topic}.",
    "Журналісти повідомили про нові досягнення у сфері {topic}."
]

authors_real = [
    "Іван Петренко", "Олена Ковальчук", "Марія Іванчук",
    "Андрій Сидоренко", "Юрій Бойко", "Наталія Гнатюк"
]

authors_fake = [
    "Anonymous", "Patriot_True", "User123", "CV19_OpenEyes",
    "RealTruth", "HiddenSource"
]

positions_real = [
    "журналіст", "репортер", "аналітик новин", "редактор"
]

positions_fake = [
    "блогер", "активіст", "користувач соцмереж", "інтернет-коментатор"
]

rows = []
n_rows = 10000  # ~7–10 МБ тексту

for _ in range(n_rows):
    topic = random.choice(topics)
    is_fake = random.random() < 0.5

    if is_fake:
        text = random.choice(fake_patterns).format(topic=topic)
        author = random.choice(authors_fake + authors_real)
        position = random.choice(positions_fake)
        rating = round(random.uniform(0.1, 0.7), 3)
    else:
        text = random.choice(real_patterns).format(topic=topic)
        author = random.choice(authors_real)
        position = random.choice(positions_real)
        rating = round(random.uniform(0.4, 0.99), 3)

    extra = " Експерти радять перевіряти джерела інформації перед поширенням новин."
    text += extra

    rows.append({
        "author": author,
        "text": text,
        "is_fake": int(is_fake),
        "rating": rating,
        "position": position
    })

with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Згенеровано файл {path} з {len(rows)} рядками")
