# Present so that `tests/ma/test_env.py` and `tests/sa/test_env.py` can share a basename.
# Without it pytest imports test modules by bare basename and the two collide at collection
# ("import file mismatch"). Added 2026-08-22 with the env2 -> env rename.
