import pandas as pd
from pathlib import Path

# ШЛЯХИ ПІД СЕБЕ ПІДРЕДАГУЙ ПРИ ПОТРЕБІ
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "liar"
OUT_CSV = ROOT / "data" / "news_with_authors_liar.csv"


def load_liar_split(path_tsv: Path | str) -> pd.DataFrame:
    """
    Завантажує один файл LIAR (train/test/valid) у DataFrame з нормальними назвами колонок.
    """
    cols = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "speaker_job_title",
        "state_info",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]
    df = pd.read_csv(path_tsv, sep="\t", header=None, names=cols)
    return df


def map_label_to_is_fake(label: str) -> int:
    """
    Мапимо 6-класовий truthfulness label у бінарний is_fake.
    0 = real, 1 = fake
    """
    label = label.strip().lower()
    real = {"true", "mostly-true", "half-true"}
    fake = {"barely-true", "false", "pants-fire"}
    if label in real:
        return 0
    if label in fake:
        return 1
    # На всякий випадок
    raise ValueError(f"Unknown label: {label}")


def map_label_to_soft_fake(label: str) -> float:
    label = label.strip().lower()
    # можна потім тюнити
    mapping = {
        "pants-fire": 1.0,
        "false": 0.9,
        "barely-true": 0.7,
        "half-true": 0.5,
        "mostly-true": 0.3,
        "true": 0.1,
    }
    return mapping[label]


def compute_speaker_ratings(df: pd.DataFrame, alpha: float = 1.0) -> pd.Series:
    """
    Обчислюємо rating для КОЖНОГО спікера на основі історії правдивості з колонок 9–13.
    Повертаємо pd.Series: index = speaker, value = rating in [0,1].
    """
    # Переконаймось, що це числа
    for col in [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Групуємо по speaker і сумуємо історію
    grouped = df.groupby("speaker")[[
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ]].sum()

    # good vs bad
    n_good = grouped["half_true_counts"] + grouped["mostly_true_counts"]
    n_bad = grouped["barely_true_counts"] + grouped["false_counts"] + grouped["pants_on_fire_counts"]

    rating = (n_good + alpha) / (n_good + n_bad + 2 * alpha)
    return rating  # index: speaker, values: float


def build_news_with_authors():
    # 1) завантажуємо train/valid/test та з’єднуємо в один DataFrame
    # train_df = load_liar_split(RAW_DIR / "train.tsv")
    # valid_df = load_liar_split(RAW_DIR / "valid.tsv")
    # test_df  = load_liar_split(RAW_DIR / "test.tsv")
    train_df = load_liar_split("data/train.tsv")
    valid_df = load_liar_split("data/valid.tsv")
    test_df  = load_liar_split("data/test.tsv")

    df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)

    # 2) is_fake
    df["is_fake"] = df["label"].apply(map_label_to_is_fake)
    df["soft_fake"] = df["label"].apply(map_label_to_soft_fake)

    # 3) rating по speaker-у
    speaker_rating = compute_speaker_ratings(df, alpha=1.0)  # Series: speaker -> rating

    # 4) будуємо author, position, text
    def make_position(row):
        job = str(row["speaker_job_title"]).strip()
        party = str(row["party_affiliation"]).strip()
        if job and job.lower() != "nan":
            return job
        if party and party.lower() != "nan":
            return f"Party: {party}"
        return "Unknown"

    def make_text(row):
        parts = []
        speaker = str(row["speaker"]).strip()
        position = make_position(row)
        statement = str(row["statement"]).strip()
        subject = str(row["subject"]).strip()
        context = str(row["context"]).strip()

        parts.append(f"Спікер: {speaker}. Посада: {position}.")
        if subject and subject.lower() != "nan":
            parts.append(f"Тема: {subject}.")
        if context and context.lower() != "nan":
            parts.append(f"Контекст: {context}.")
        parts.append(f"Заява: {statement}")
        return " ".join(parts)

    df["author"] = df["speaker"].astype(str)
    df["position"] = df.apply(make_position, axis=1)
    df["text"] = df.apply(make_text, axis=1)

    # 5) додаємо числовий rating (якщо для спікера немає історії – 0.5)
    df["rating"] = df["author"].map(speaker_rating).fillna(0.5)

    # 6) формуємо фінальний датафрейм у твоєму форматі
    out_df = df[["author", "text", "is_fake", "soft_fake", "rating", "position"]].copy()

    # 7) зберігаємо
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    build_news_with_authors()
