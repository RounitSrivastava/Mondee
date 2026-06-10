import ast

valid = 0
invalid = 0

with open(
    "data/meta_Electronics.json",
    encoding="utf-8"
) as f:

    for line in f:

        try:
            ast.literal_eval(
                line.strip()
            )
            valid += 1

        except Exception:
            invalid += 1

print(
    f"Valid records   : {valid}"
)

print(
    f"Invalid records : {invalid}"
)