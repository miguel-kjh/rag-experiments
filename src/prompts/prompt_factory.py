import json
from pathlib import Path

class PromptFactory:
    def __init__(self, json_path: str = "src/prompts/prompts.json"):
        self.prompts = self._load_prompts(json_path)

    def _load_prompts(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_prompts(self, task: str, lang: str = "en"):
        """
        Retorna el system prompt y user prompt adaptado para la tarea e idioma especificados.
        """
        if task not in self.prompts:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(self.prompts.keys())}")
        if lang not in self.prompts[task]:
            raise ValueError(f"Unsupported language '{lang}' for task '{task}'. Use 'en' or 'es'.")

        system_prompt = self.prompts[task][lang]["system"]
        user_prompt = self.prompts[task][lang]["user"]
        return system_prompt, user_prompt


if __name__ == "__main__":
    pf = PromptFactory()
    system, user = pf.get_prompts("query_rewriter", lang="es", query="¿Qué es la inteligencia artificial?")
    print("System Prompt:\n", system)
    print("\nUser Prompt:\n", user)