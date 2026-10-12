"""Create 1000 synthetic documents for reproducible functional benchmarking.

NOTE: This is demonstration data, NOT a real research or reading corpus.
"""

from pathlib import Path

PARAGRAPHS = [
    "Python дозволяє працювати з текстом. Генератори повертають значення по одному. "
    "Ітератори допомагають зменшити споживання пам'яті. ",
    "Алгоритми пошуку використовують індекс і словник термінів. "
    "Слова різних мов потребують Unicode нормалізації NFC. ",
    "Документи містять слова, цифри 12345, апострофи п'ять, don't та дефіси веб-пошук. "
    "Обробка потоку означає послідовне читання файлів. ",
    "Information retrieval is about documents, tokens, search, and ranking. "
    "A generator is lazy; a list is eager. ",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "demo"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        text = f"Навчальний документ {i:04d}. Тема: Python і пошукові системи.\n"
        text += (PARAGRAPHS[i % len(PARAGRAPHS)] * 30 + "\n") * 3
        (root / f"document_{i:04d}.txt").write_text(text, encoding="utf-8")
    print(f"Створено 1000 демонстраційних документів у {root}")


if __name__ == "__main__":
    main()
