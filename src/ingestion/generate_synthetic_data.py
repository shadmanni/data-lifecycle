"""
Generates synthetic 'source code' files across 3 versions, with realistic
version-to-version redundancy (some files unchanged, some lightly edited,
some new, some removed). No internet download needed at all.
"""
import os
import random

random.seed(42)

WORDS = ["int", "void", "return", "struct", "char", "float", "if", "else", "for",
         "while", "static", "const", "define", "include", "typedef", "malloc",
         "free", "printf", "buffer", "size", "index", "value", "config", "handler"]


def make_function_block(seed_words):
    lines = []
    for _ in range(random.randint(5, 15)):
        line = " ".join(random.choices(seed_words, k=random.randint(3, 8))) + ";"
        lines.append(line)
    return "\n".join(lines)


def make_file_content(num_blocks=6):
    return "\n\n".join(make_function_block(WORDS) for _ in range(num_blocks))


def edit_content(content, edit_intensity=0.2):
    """Simulates realistic small edits: insert/delete/modify a fraction of lines."""
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        r = random.random()
        if r < edit_intensity * 0.3:
            continue  # delete line
        elif r < edit_intensity * 0.6:
            new_lines.append(line)
            new_lines.append(make_function_block(WORDS).split("\n")[0])  # insert line
        elif r < edit_intensity:
            new_lines.append(" ".join(random.choices(WORDS, k=len(line.split()))) + ";")  # modify line
        else:
            new_lines.append(line)  # unchanged
    return "\n".join(new_lines)


def generate_version_set(base_dir="data/raw", num_files=200, num_versions=3):
    os.makedirs(base_dir, exist_ok=True)

    # v1: all new files
    v1_dir = os.path.join(base_dir, "synth_v1")
    os.makedirs(v1_dir, exist_ok=True)
    file_contents = {}
    for i in range(num_files):
        fname = f"module_{i}.c"
        content = make_file_content()
        file_contents[fname] = content
        with open(os.path.join(v1_dir, fname), "w") as f:
            f.write(content)
    print(f"Created {num_files} files in {v1_dir}")

    # subsequent versions: most files carry over with small edits,
    # a few files are removed, a few new files are added
    prev_contents = file_contents
    for v in range(2, num_versions + 1):
        v_dir = os.path.join(base_dir, f"synth_v{v}")
        os.makedirs(v_dir, exist_ok=True)
        new_contents = {}

        for fname, content in prev_contents.items():
            if random.random() < 0.05:
                continue  # 5% of files removed this version
            edited = edit_content(content, edit_intensity=random.choice([0.0, 0.1, 0.3]))
            new_contents[fname] = edited
            with open(os.path.join(v_dir, fname), "w") as f:
                f.write(edited)

        # add a handful of new files each version
        for i in range(15):
            fname = f"module_new_v{v}_{i}.c"
            content = make_file_content()
            new_contents[fname] = content
            with open(os.path.join(v_dir, fname), "w") as f:
                f.write(content)

        print(f"Created {len(new_contents)} files in {v_dir}")
        prev_contents = new_contents


if __name__ == "__main__":
    generate_version_set()