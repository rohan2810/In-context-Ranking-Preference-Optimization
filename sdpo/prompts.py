prompts = {
"meta-llama/Llama-3.2-3B-Instruct": """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system).<|eot_id|><|start_header_id|>user<|end_header_id|>

Based on the conversation, you reply me with 20 recommendations without extra sentences. Here is the conversation: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


""",

"meta-llama/Llama-3.2-1B-Instruct": """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system).<|eot_id|><|start_header_id|>user<|end_header_id|>

Based on the conversation, you reply me with 20 recommendations without extra sentences. Here is the conversation: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


""",

"google/gemma-2-2b-it": """
<start_of_turn>user
Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. Here is the conversation: {}<end_of_turn>
<start_of_turn>model

""",

"microsoft/Phi-3-mini-4k-instruct": """
<|system|>
Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system).<|end|>
<|user|>
Based on the conversation, you reply me with 20 recommendations without extra sentences. Here is the conversation: {}<|end|>
<|assistant|>

"""
}
