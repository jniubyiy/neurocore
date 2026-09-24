# merge_code_smart.py
import os
from pathlib import Path

# ============================================================
# 1. Корневая директория проекта
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR   # скрипт лежит в корне проекта

# ============================================================
# 2. Папки, которые ПОЛНОСТЬЮ ПРОПУСКАЕМ (не сканируются)
# ------------------------------------------------------------
# Правила сопоставления:
#   - "target"          -> любая папка с таким именем на любом уровне
#   - "src/layers/mamba"-> точное совпадение ИЛИ суффикс на границе сегмента
# При пропуске папки её подпапки тоже не сканируются.
# ============================================================
EXCLUDE_DIR_PATTERNS = [
    "target",
    ".git",
    ".vscode",
    "__pycache__",
]

# ============================================================
# 3. Папки, которые НЕ СКАНИРУЮТСЯ, но УПОМИНАЮТСЯ в отчёте
# ------------------------------------------------------------
# Те же правила сопоставления, что и выше.
# ============================================================
MENTIONED_EXCLUDE_DIR_PATTERNS = [
    "debug",
    "release",
    "layers/adaptive_activation",
    "layers/adaptive_dropout",
    "layers/adaptive_normalization",
    "layers/batch_renorm",
    "layers/combiner",
    "layers/combiner_connector",
    "layers/concrete_dropout",
    "layers/dual_anchor",
    "layers/dual_slope_relu",
    "layers/feature_fusion",
    "layers/identity",
    "layers/ind_rnn",
    "layers/layers_special",
    "layers/learnable_mish",
    "layers/learnable_softplus",
    "layers/linear_attention",
    "layers/mamba",
    "layers/memory",
    "layers/multi_resolution_kan_linear",
    "layers/per_feature_attention",
    "layers/relative_position_attention",
    "layers/rms_norm_learnable_eps",
    "layers/sparse_feature_selection_gate",
    "layers/spectral_norm_linear",
]

# ============================================================
# 4. Количество частей, на которые нужно разбить итоговый файл
# ============================================================
PARTS = 15

# Базовое имя выходных файлов (без расширения)
BASE_OUTPUT_NAME = "merged_project_code"

# Разделители
SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80

# Сколько ПОСЛЕДНИХ логов показывать в консольной диагностике
LOG_TAIL = 32

# Файлы, исключаемые из сборки (автоматически: сам скрипт и выходные файлы)
EXCLUDE_FILES = set()


def matches_dir_pattern(rel_dir: Path, pattern: str) -> bool:
    """
    Сопоставление папки с шаблоном.

      * шаблон без '/' — совпадение по имени сегмента на любой глубине;
      * шаблон с '/' — совпадение пути ЦЕЛИКОМ либо по суффиксу на границе
        сегмента: "layers/mamba" матчит "layers/mamba", "src/layers/mamba",
        "a/b/layers/mamba", но не "my_layers/mamba_x".
    """
    rel_posix = rel_dir.as_posix().replace("\\", "/").strip("/")
    pat = pattern.replace("\\", "/").strip("/")

    if "/" not in pat:
        return rel_dir.name.lower() == pat.lower()

    rp = rel_posix.lower()
    pp = pat.lower()
    return rp == pp or rp.endswith("/" + pp)


def collect_files(root_dir, exclude_patterns, mentioned_patterns):
    """
    Собирает файлы проекта: Cargo.toml, .rs, .html, .js, .css, .comp, .bat.
    Возвращает:
      * список (тип, отн.путь, полный путь);
      * множество найденных упомянутых каталогов (для шапки отчёта);
      * список ВСЕХ путей, попавших под EXCLUDE (для консольного лога);
      * список ВСЕХ путей, попавших под MENTIONED (для консольного лога).
    """
    root = Path(root_dir).resolve()
    collected = []
    found_mentioned = set()
    excluded_hits = []   # копим все срабатывания, потом покажем хвост
    mentioned_hits = []

    for current_dir, dirs, filenames in os.walk(root):
        current_path = Path(current_dir).resolve()
        try:
            rel_current = current_path.relative_to(root)
        except ValueError:
            rel_current = Path(".")

        new_dirs = []
        for d in dirs:
            rel_sub = (rel_current / d) if rel_current != Path(".") else Path(d)

            hit_exclude = next(
                (p for p in exclude_patterns
                 if matches_dir_pattern(rel_sub, p)),
                None,
            )
            hit_mention = next(
                (p for p in mentioned_patterns
                 if matches_dir_pattern(rel_sub, p)),
                None,
            )

            if hit_exclude is not None:
                # Только в консольный лог, в файлы это не попадает.
                excluded_hits.append(
                    f"{rel_sub.as_posix()}  <- правило: {hit_exclude!r}"
                )
                continue

            if hit_mention is not None:
                found_mentioned.add(rel_sub.as_posix())
                mentioned_hits.append(
                    f"{rel_sub.as_posix()}  <- правило: {hit_mention!r}"
                )
                continue

            new_dirs.append(d)
        dirs[:] = new_dirs

        for fname in filenames:
            full_path = (Path(current_dir) / fname).resolve()
            if full_path in EXCLUDE_FILES:
                continue

            fname_lower = fname.lower()
            if fname_lower == "cargo.toml":
                ftype = "toml"
            elif fname_lower.endswith(".rs"):
                ftype = "rs"
            elif fname_lower.endswith(".html"):
                ftype = "html"
            elif fname_lower.endswith(".js"):
                ftype = "js"
            elif fname_lower.endswith(".css"):
                ftype = "css"
            elif fname_lower.endswith(".comp"):
                ftype = "comp"
            elif fname_lower.endswith(".bat"):
                ftype = "bat"
            else:
                continue

            try:
                rel_path = full_path.relative_to(root)
            except ValueError:
                rel_path = full_path

            collected.append((ftype, str(rel_path), full_path))

    type_order = {"toml": 0, "rs": 1, "html": 2, "js": 3,
                  "css": 4, "comp": 5, "bat": 6}
    collected.sort(key=lambda x: (type_order.get(x[0], 99), x[1]))

    return collected, found_mentioned, excluded_hits, mentioned_hits


def count_lines_of_file(full_path):
    """Безопасно подсчитывает количество строк в файле. Возвращает 0 при ошибке."""
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось прочитать {full_path} "
              f"для подсчёта строк: {e}")
        return 0


def distribute_files_by_lines(file_infos, num_parts):
    """
    Жадно распределяет файлы по num_parts корзинам так, чтобы суммарное число
    строк в каждой корзине было как можно более равномерным.
    """
    sorted_infos = sorted(file_infos, key=lambda x: x[3], reverse=True)

    parts = [[] for _ in range(num_parts)]
    sums = [0] * num_parts

    for info in sorted_infos:
        min_idx = min(range(num_parts), key=lambda i: sums[i])
        parts[min_idx].append(info)
        sums[min_idx] += info[3]

    return parts


def merge_files(entries, output_path, mentioned_dirs_found,
                part_num=None, total_parts=None, total_lines=None):
    """Записывает содержимое файлов в выходной текстовый файл."""
    processed = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(SEPARATOR + "\n")
        out.write("СБОРКА КОДА ПРОЕКТА\n")
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

    print(f"Часть {part_num}: обработано {processed} файлов, "
          f"пропущено {skipped} -> {output_path}")


def write_partition_message(file_obj, part_num, total_parts, is_start):
    """Вставляет сообщение о переходе между частями."""
    if is_start:
        if part_num == 2 and total_parts == 2:
            msg = "Это начало второй половины."
        else:
            msg = f"Это начало части {part_num}."
    else:
        msg = f"Это конец части {part_num}, дождись части {part_num + 1}."
    file_obj.write("\n" + SEPARATOR + "\n")
    file_obj.write(msg + "\n")
    file_obj.write(SEPARATOR + "\n\n")


# ============================================================
# Консольная диагностика: последние N срабатываний
# ============================================================

def print_exclude_log(hits, patterns, tail=LOG_TAIL):
    print()
    print(SEPARATOR)
    print(f"EXCLUDE_DIR_NAMES — последние {tail} срабатываний")
    print(f"Задано шаблонов: {len(patterns)} "
          f"-> {', '.join(repr(p) for p in patterns)}")
    print(SEPARATOR)
    if not hits:
        print("  (срабатываний не было)")
        return
    shown = hits[-tail:]
    print(f"  Всего срабатываний: {len(hits)}. "
          f"Показаны последние {len(shown)}:")
    for line in shown:
        print(f"  [X] {line}")


def print_mentioned_log(hits, patterns, tail=LOG_TAIL):
    print()
    print(SEPARATOR)
    print(f"MENTIONED_EXCLUDE_DIRS — последние {tail} срабатываний")
    print(f"Задано шаблонов: {len(patterns)}")
    print(SEPARATOR)
    if not hits:
        print("  (срабатываний не было)")
        return
    shown = hits[-tail:]
    print(f"  Всего срабатываний: {len(hits)}. "
          f"Показаны последние {len(shown)}:")
    for line in shown:
        print(f"  [M] {line}")


# ============================================================

if __name__ == "__main__":
    this_script = Path(__file__).resolve()
    EXCLUDE_FILES = {this_script}
    for p in range(1, PARTS + 1):
        output_file = Path(f"{BASE_OUTPUT_NAME}_part{p}.txt").resolve()
        EXCLUDE_FILES.add(output_file)
    EXCLUDE_FILES.add(Path(f"{BASE_OUTPUT_NAME}.txt").resolve())

    entries, mentioned_found, excluded_hits, mentioned_hits = collect_files(
        SOURCE_DIR,
        EXCLUDE_DIR_PATTERNS,
        MENTIONED_EXCLUDE_DIR_PATTERNS,
    )

    # --- Логи только в консоль, в файлы не пишутся ---
    print_exclude_log(excluded_hits, EXCLUDE_DIR_PATTERNS, LOG_TAIL)
    print_mentioned_log(mentioned_hits, MENTIONED_EXCLUDE_DIR_PATTERNS, LOG_TAIL)

    if not entries:
        print("Не найдено ни одного подходящего файла "
              "(Cargo.toml, .rs, .html, .js, .css, .comp, .bat) "
              "с учётом исключений.")
        exit(1)

    file_infos = []
    for ftype, rel_path, full_path in entries:
        line_cnt = count_lines_of_file(full_path)
        file_infos.append((ftype, rel_path, full_path, line_cnt))
        print(f"[INFO] {rel_path}: {line_cnt} строк(и)")

    if PARTS <= 1:
        output_path = f"{BASE_OUTPUT_NAME}.txt"
        total_lines_all = sum(info[3] for info in file_infos)
        merge_files(
            [(info[0], info[1], info[2]) for info in file_infos],
            output_path,
            mentioned_found,
            total_lines=total_lines_all,
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