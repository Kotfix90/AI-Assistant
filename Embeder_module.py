import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional

class Embedder:
    def __init__(self, model_name: str = "ai-forever/ru-en-RoSBERTa"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Использование устройства: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.vector_size = self._get_vector_size()

    def _get_vector_size(self) -> int:
        """Автоматически определяет размерность вектора модели."""
        test_embed = self.get_embedding("test")
        return len(test_embed)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str) -> List[float]:
        """Генерирует L2-нормализованный вектор для текста."""
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded)
            embedding = self._mean_pooling(model_output, encoded["attention_mask"])
            
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding[0].cpu().numpy().tolist()