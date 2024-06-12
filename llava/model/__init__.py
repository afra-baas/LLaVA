try:
    # # Set this variable based on your requirements
    # use_llama_method2 = False
    # print('use_llama_method2', use_llama_method2)
    # if use_llama_method2:
    #     from .language_model.llava_llama_method2 import LlavaLlamaForCausalLM, LlavaConfig
    # else:
    #     from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

    # from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    # from .language_model.llava_llama_method2 import LlavaLlamaForCausalLM, LlavaConfig

    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
except:
    pass
