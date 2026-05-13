FEATURE_COLUMNS: list[str] = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
]

TARGET_COLUMN = "medv"

ALL_COLUMNS: list[str] = FEATURE_COLUMNS + [TARGET_COLUMN]
