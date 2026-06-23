# merge_rust_code_smart.py
import os
import math
from pathlib import Path

# ============================================================
# 1. Корневая директория проекта
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR   # скрипт лежит в корне neurocore

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


def count_lines_of_file(full_path):
    """Безопасно подсчитывает количество строк в файле. Возвращает 0 при ошибке."""
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось прочитать {full_path} для подсчёта строк: {e}")
        return 0


def distribute_files_by_lines(file_infos, num_parts):
    """
    Жадно распределяет файлы по num_parts корзинам так, чтобы суммарное число строк
    в каждой корзине было как можно более равномерным.
    Возвращает список корзин, каждая корзина — список элементов file_info.
    """
    sorted_infos = sorted(file_infos, key=lambda x: x[3], reverse=True)

    parts = [[] for _ in range(num_parts)]
    sums = [0] * num_parts

    for info in sorted_infos:
        min_idx = min(range(num_parts), key=lambda i: sums[i])
        parts[min_idx].append(info)
        sums[min_idx] += info[3]

    return parts


def merge_files(entries, output_path, mentioned_dirs_found, part_num=None, total_parts=None, total_lines=None):
    """Записывает содержимое файлов в выходной текстовый файл."""
    processed = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(SEPARATOR + "\n")
        out.write("СБОРКА КОДА ПРОЕКТА neurocore\n")
        out.write(f"Корень проекта: {Path(SOURCE_DIR).resolve()}\n")
        if total_parts and total_parts > 1:
            out.write(f"Часть {part_num} из {total_parts}\n")
        if total_lines is not None:
            out.write(f"Общее количество строк в этой части: {total_lines}\n")
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

        # Статистика больше не пишется в файл

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
        # Всегда используем универсальное сообщение
        msg = f"Это конец части {part_num}, дождись части {part_num + 1}."
    file_obj.write("\n" + SEPARATOR + "\n")
    file_obj.write(msg + "\n")
    file_obj.write(SEPARATOR + "\n\n")


if __name__ == "__main__":
    this_script = Path(__file__).resolve()
    EXCLUDE_FILES = {this_script}
    for p in range(1, PARTS + 1):
        output_file = Path(f"{BASE_OUTPUT_NAME}_part{p}.txt").resolve()
        EXCLUDE_FILES.add(output_file)
    EXCLUDE_FILES.add(Path(f"{BASE_OUTPUT_NAME}.txt").resolve())

    entries, mentioned_found = collect_rust_files(
        SOURCE_DIR,
        EXCLUDE_DIR_NAMES,
        MENTIONED_EXCLUDE_DIRS
    )

    if not entries:
        print("Не найдено ни одного Cargo.toml или .rs файла (с учётом исключений).")
        exit(1)

    file_infos = []
    for ftype, rel_path, full_path in entries:
        line_cnt = count_lines_of_file(full_path)
        file_infos.append((ftype, rel_path, full_path, line_cnt))
        print(f"[INFO] {rel_path}: {line_cnt} строк(и)")

    total_files = len(file_infos)

    if PARTS <= 1:
        output_path = f"{BASE_OUTPUT_NAME}.txt"
        total_lines_all = sum(info[3] for info in file_infos)
        merge_files(
            [(info[0], info[1], info[2]) for info in file_infos],
            output_path,
            mentioned_found,
            total_lines=total_lines_all
        )
    else:
        distributed_parts = distribute_files_by_lines(file_infos, PARTS)

        for part_idx, part_entries in enumerate(distributed_parts, start=1):
            part_triples = [(info[0], info[1], info[2]) for info in part_entries]
            part_lines = sum(info[3] for info in part_entries)

            output_path = f"{BASE_OUTPUT_NAME}_part{part_idx}.txt"
            merge_files(part_triples, output_path, mentioned_found,
                        part_num=part_idx, total_parts=PARTS,
                        total_lines=part_lines)

            if part_idx < PARTS:
                with open(output_path, "a", encoding="utf-8") as f:
                    write_partition_message(f, part_idx, PARTS, is_start=False)
            if part_idx > 1:
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                with open(output_path, "w", encoding="utf-8") as f:
                    write_partition_message(f, part_idx, PARTS, is_start=True)
                    f.write(content)