# unsloth-multi-gpu-vision

This is an extension from [cwpeng's](https://github.com/cwpeng-cn/unsloth-multi-gpu), you can now train Qwen3-VL on multi-GPU:  
1. For the first run, you need to disable multi-gpu for Unsloth to compile into unsloth_compiled_cache   
2. After that, add os.environ["UNSLOTH_COMPILE_DISABLE"] = "1", disable Unsloth compilation to avoid hanging <= I don't know why, this will reduce speed but my experiment is too small for noticeable effect     
3. The root cause seems to be related to gradient checkpointing  
4. Can run with DeepSpeed.  
5. Working with GRPO for VL model.  
