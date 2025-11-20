import os
import random
import numpy as np
import torch

FOLDER_DATA      = "data"
FOLDER_RAW       = os.path.join(FOLDER_DATA, "raw")
FOLDER_PROCESSED = os.path.join(FOLDER_DATA, "processed")
FOLDER_COMBINED  = os.path.join(FOLDER_DATA, "combined")
FOLDER_DB        = os.path.join(FOLDER_DATA, "db")

RAGBENCH_SUBSETS = [
    'covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa'
]

SEED = 123

SYSTEM_PROMPT_RAG = "You are a helpful AI assistant. Use the provided context to answer questions."
SYSTEM_PROMPT_KNOWLEDGE = """
You are an expert at answering questions. Given the question, provide a concise and accurate answer based on your knowledge.
"""
PROMPT = """Answer the question based only on the following context:
Context: {context}

Question: {question}

Answer:"""

SYSTEM_PROMPT_FOR_GENERATE_ENTITIES = """
As a knowledge analyzer, your task is to dissect and understand an article provided by the user. You are required to perform the following steps:
1. Summarize the Article: Provide a concise summary of the entire article, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the article. These entities should include but not limited to:
    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article's discussion.

Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively. Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.

Here is the format you should use for your response:

{
  "summary":  "<A concise summary of the article>",
  "entities": ["entity1", "entity2", ...]
}
"""

SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves two main objectives:
1. Rephrasing Content: The user will identify two specific entities mentioned in the article. You are required to rephrase the content of the article twice:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of Interaction between <entity1> and <entity2> in context of <title>
<Discussion on how the two entities interact within the article>
"""

SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves three main objectives:
1. Rephrasing Content: The user will identify three specific entities mentioned in the article. You are required to rephrase the content of the article three times:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
    * Lastly, emphasizing the third entity.
2. Analyzing Interactions: Discuss how these three specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of <title> in relation to <entity3>
<Rephrased content focusing on the third entity>

### Discussion of Interaction between <entity1>, <entity2> and <entity3> in context of <title>
<Discussion on how the three entities interact within the article>
"""

def seed_everything(seed):
    """Set seed for reproducibility."""
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Environment setup
# -----------------------------
def setup_environment():
    """Load environment variables and set the OpenAI API key."""
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# -----------------------------
# Builder prompts
# -----------------------------
def build_prompt_query_expander(tokenizer, system_prompt: str, prompt: str, query: str, enable_thinking: bool = False) -> str:
    """Builds the chat prompt for a single example using the tokenizer chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt.format(query=query)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def main():
    # create folders if they don't exist
    for folder in [FOLDER_DATA, FOLDER_RAW, FOLDER_PROCESSED, FOLDER_COMBINED, FOLDER_DB]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

if __name__ == "__main__":
    main()

