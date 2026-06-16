# merge_rust_code_smart.py
import os
import math
from pathlib import Path

# ============================================================
# 1. Корневая директория проекта
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR   # скрипт лежит в корне neurocore

# Если нужно указать жёстко:
# SOURCE_DIR = r"D:\neurocore"

# ============================================================
# 2. Папки, которые ПОЛНОСТЬЮ ПРОПУСКАЕМ (не сканируются)
# ============================================================
EXCLUDE_DIR_NAMES = [
    "target",
    ".git",
    ".vscode",
    "__pycache__",
]

# ============================================================
# 3. Папки, которые НЕ СКАНИРУЮТСЯ, но УПОМИНАЮТСЯ в отчёте
# ============================================================
MENTIONED_EXCLUDE_DIRS = [
    "debug",
    "release",
]

# ============================================================
# 4. Количество частей, на которые нужно разбить итоговый файл
#    (если PARTS = 1, разбивка не производится)
# ============================================================
PARTS = 3

# Базовое имя выходных файлов (без расширения)
BASE_OUTPUT_NAME = "merged_neurocore_code"

# Разделители
SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80

# Файлы, исключаемые из сборки (автоматически: сам скрипт и выходные файлы)
EXCLUDE_FILES = set()


def collect_rust_files(root_dir, exclude_dirs, mentioned_dirs):
    """Собирает Cargo.toml и все .rs файлы."""
    root = Path(root_dir).resolve()
    collected = []
    found_mentioned = set()

    for current_dir, dirs, filenames in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in exclude_dirs and d not in mentioned_dirs
        ]

        for d in os.listdir(current_dir):
            if d in mentioned_dirs and (Path(current_dir) / d).is_dir():
                found_mentioned.add(d)

        for fname in filenames:
            full_path = (Path(current_dir) / fname).resolve()
            if full_path in EXCLUDE_FILES:
                continue

            if fname.lower() == "cargo.toml":
                ftype = "toml"
            elif fname.lower().endswith(".rs"):
                ftype = "rs"
            else:
                continue

            try:
                rel_path = full_path.relative_to(root)
            except ValueError:
                rel_path = full_path

            collected.append((ftype, str(rel_path), full_path))

    tomls = sorted([c for c in collected if c[0] == "toml"], key=lambda x: x[1])
    sources = sorted([c for c in collected if c[0] == "rs"], key=lambda x: x[1])
    return tomls + sources, found_mentioned


def merge_files(entries, output_path, mentioned_dirs_found, part_num=None, total_parts=None):
    """Записывает содержимое файлов в выходной текстовый файл."""
    processed = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(SEPARATOR + "\n")
        out.write("СБОРКА КОДА ПРОЕКТА neurocore\n")
        out.write(f"Корень проекта: {Path(SOURCE_DIR).resolve()}\n")
        if total_parts and total_parts > 1:
            out.write(f"Часть {part_num} из {total_parts}\n")
        out.write(SEPARATOR + "\n")

        if mentioned_dirs_found:
            out.write("\nИсключены из поиска (упомянутые каталоги):\n")
            for d in sorted(mentioned_dirs_found):
                out.write(f"  - {d}\n")
            out.write(SEPARATOR + "\n")

        out.write("\n")

        for idx, (ftype, rel_path, full_path) in enumerate(entries, start=1):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                print(f"[ОШИБКА] Не удалось прочитать {rel_path}: {e}")
                out.write(SEPARATOR + "\n")
                out.write(f"ФАЙЛ {idx}: {rel_path}\n")
                out.write(f"Полный путь: {full_path}\n")
                out.write(f"Тип: {ftype}\n")
                out.write(f"!!! ОШИБКА ЧТЕНИЯ: {e} !!!\n")
                out.write(SEPARATOR + "\n\n")
                skipped += 1
                continue

            if not code.strip():
                code = "<!-- Файл пуст -->"

            out.write(SEPARATOR + "\n")
            out.write(f"ФАЙЛ {idx}: {rel_path}\n")
            out.write(f"Полный путь: {full_path}\n")
            out.write(f"Тип: {ftype}\n")
            out.write(SUB_SEPARATOR + "\n")
            out.write(code.rstrip() + "\n")
            out.write(SEPARATOR + "\n")
            out.write(f"КОНЕЦ ФАЙЛА: {rel_path}\n")
            out.write(SEPARATOR + "\n\n")

            print(f"[OK] Добавлен: {rel_path}")
            processed += 1

        out.write(SEPARATOR + "\n")
        out.write(f"Обработано успешно: {processed}\n")
        out.write(f"Пропущено / с ошибками: {skipped}\n")
        out.write(SEPARATOR + "\n")

    print(f"Часть {part_num}: обработано {processed} файлов, пропущено {skipped} -> {output_path}")


def write_partition_message(file_obj, part_num, total_parts, is_start):
    """Вставляет сообщение о переходе между частями."""
    msg = ""
    if is_start:
        if part_num == 2:
            msg = "Это начало второй половины."
        else:
            msg = f"Это начало части {part_num}."
    else:
        if part_num == total_parts - 1:
            msg = f"Это конец первой половины, дождись второй половины."
        else:
            msg = f"Это конец части {part_num}, дождись части {part_num + 1}."
    file_obj.write("\n" + SEPARATOR + "\n")
    file_obj.write(msg + "\n")
    file_obj.write(SEPARATOR + "\n\n")


if __name__ == "__main__":
    this_script = Path(__file__).resolve()
    # Исключим все возможные выходные файлы (если PARTS=2, то два файла)
    EXCLUDE_FILES = {this_script}
    for p in range(1, PARTS + 1):
        output_file = Path(f"{BASE_OUTPUT_NAME}_part{p}.txt").resolve()
        EXCLUDE_FILES.add(output_file)
    # Добавим и вариант с одной частью на всякий случай
    EXCLUDE_FILES.add(Path(f"{BASE_OUTPUT_NAME}.txt").resolve())

    entries, mentioned_found = collect_rust_files(
        SOURCE_DIR,
        EXCLUDE_DIR_NAMES,
        MENTIONED_EXCLUDE_DIRS
    )

    if not entries:
        print("Не найдено ни одного Cargo.toml или .rs файла (с учётом исключений).")
        exit(1)

    total_files = len(entries)
    if PARTS <= 1:
        # Одна часть – сохраняем как раньше
        output_path = f"{BASE_OUTPUT_NAME}.txt"
        merge_files(entries, output_path, mentioned_found)
    else:
        # Разбиваем на части
        files_per_part = math.ceil(total_files / PARTS)
        for part in range(1, PARTS + 1):
            start_idx = (part - 1) * files_per_part
            end_idx = min(start_idx + files_per_part, total_files)
            part_entries = entries[start_idx:end_idx]
            output_path = f"{BASE_OUTPUT_NAME}_part{part}.txt"
            merge_files(part_entries, output_path, mentioned_found, part_num=part, total_parts=PARTS)
            # Добавляем сообщение в конце каждой части, кроме последней
            if part < PARTS:
                with open(output_path, "a", encoding="utf-8") as f:
                    write_partition_message(f, part, PARTS, is_start=False)
            # И сообщение в начале каждой части, кроме первой
            if part > 1:
                # Чтобы вставить в начало, прочитаем файл и перезапишем с сообщением в начале
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                with open(output_path, "w", encoding="utf-8") as f:
                    write_partition_message(f, part, PARTS, is_start=True)
                    f.write(content)