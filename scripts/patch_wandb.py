import wandb.sdk.lib.apikey as k

file_path = k.__file__
with open(file_path, 'r') as f:
    content = f.read()

target = "if len(suffix) != 40:"
if target in content:
    content = content.replace("if len(suffix) != 40:", "if False and len(suffix) != 40:")
    with open(file_path, 'w') as f:
        f.write(content)
    print("SUCCESS: WandB 40-character key length restriction removed!")
else:
    print("WandB key length check already removed or target not found.")
