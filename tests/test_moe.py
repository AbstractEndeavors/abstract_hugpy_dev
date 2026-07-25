from imports.src.abstract_hugpy_dev.managers.spill import _gguf_scan_moe
from imports.src.abstract_hugpy_dev.managers.spill import *
from abstract_utilities import *
models_path ="/mnt/llm_storage/models/"
dirs,ggufs = get_files_and_dirs(models_path,allowed_exts='.gguf')
for gguf in ggufs:
    
    result = gguf_moe_detail(gguf)
    if result.get('is_moe'):
        print(gguf)
        print(result)
