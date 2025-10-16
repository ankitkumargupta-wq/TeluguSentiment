
import torch, re, json, emoji
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

class EnhancedTeluguPreprocessor:
    def __init__(self, rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
        self.translit_variants = self.rules.get("translit_variants", {})
        self.negations = self.rules.get("negation_words", {})
        self.boosters = self.rules.get("booster_words", [])
        self.punctuation_pattern = re.compile(r"[^\w\s]", re.UNICODE)

    def _apply_rules(self, text, mapping):
        for key, val in mapping.items():
            text = re.sub(rf"\\b{key}\\b", val, text, flags=re.IGNORECASE)
        return text

    def _normalize_emoji(self, text):
        for char in text:
            if char in emoji.EMOJI_DATA:
                desc = emoji.demojize(char)
                if any(pos in desc for pos in ["smile","joy","heart","thumbsup","clap","tada","pray"]):
                    text += " positive"
                elif any(neg in desc for neg in ["angry","sad","thumbsdown","cry","frown","rage"]):
                    text += " negative"
        return text

    def preprocess(self, text):
        text = text.strip().lower()
        text = self._apply_rules(text, self.translit_variants)
        text = self._normalize_emoji(text)
        text = self.punctuation_pattern.sub("", text)
        return text

class MuRILSentiment:
    def __init__(self, model_name="DSL-13-SRMAP/MuRIL_WR", rules_path=None):
        import os
        import TeluguSentiment

        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(TeluguSentiment.__file__), "rules.json")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.preprocessor = EnhancedTeluguPreprocessor(rules_path)
        self.labels = ["negative","neutral","positive"]
        self.label_emojis = {"negative":"😞","neutral":"😐","positive":"😊"}

    def _contains_telugu(self, text):
        return bool(re.search(r'[\\u0C00-\\u0C7F]', text))

    def predict(self, text):
        if not isinstance(text,str) or not text.strip():
            return "NEUTRAL 😐", 0.0, 0.0

        processed_text = text.strip() if self._contains_telugu(text) else self.preprocessor.preprocess(text)
        inputs = self.tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
        sentiment = self.labels[pred_idx]
        emoji_label = self.label_emojis[sentiment]
        confidence = probs[pred_idx]*100
        return f"{sentiment.upper()} {emoji_label}", round(probs[pred_idx]*100,2), round(confidence,2)

    def predict_excel(self, file_path, output_path=None):
        import pandas as pd
        df = pd.read_excel(file_path)
        if "clean_text" not in df.columns:
            raise ValueError("Excel must contain a 'clean_text' column.")

        df = df.copy()
        df = df.dropna(subset=["clean_text"])
        labels, scores, confidences = [], [], []

        for text in df["clean_text"]:
            label, score, conf = self.predict(str(text))
            labels.append(label)
            scores.append(score)
            confidences.append(conf)

        df["sentiment_label"] = labels
        df["sentiment_score"] = scores
        df["sentiment_confidence"] = confidences

        output_file = output_path if output_path else "sentiment_results.xlsx"
        df.to_excel(output_file, index=False)
        return output_file
