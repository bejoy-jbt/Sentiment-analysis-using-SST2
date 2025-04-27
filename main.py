import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name: The name of the pre-trained model to use (default: DistilBERT finetuned for SST-2)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    def analyze(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[Dict]:
        """
        Analyze sentiment of the provided text(s).
        
        Args:
            texts: A single text string or a list of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            List of dictionaries containing sentiment analysis results
        """
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self._analyze_batch(batch_texts)
            results.extend(batch_results)
            
        return results
    
    def _analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Process a batch of texts and return sentiment analysis results."""
        # Tokenize texts
        encoded_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            
        # Process predictions
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = scores.cpu().numpy()
        
        results = []
        for i, text in enumerate(texts):
            # Assuming binary sentiment (positive/negative)
            negative_score = float(scores[i, 0])
            positive_score = float(scores[i, 1])
            predicted_class = "positive" if positive_score > negative_score else "negative"
            
            results.append({
                "text": text,
                "sentiment": predicted_class,
                "confidence": max(positive_score, negative_score),
                "scores": {
                    "negative": negative_score,
                    "positive": positive_score
                }
            })
            
        return results

    def visualize_result(self, result):
        """
        Create a visualization of the sentiment analysis result.
        
        Args:
            result: A single result dictionary from the analyze method
        """
        labels = ['Negative', 'Positive']
        scores = [result['scores']['negative'], result['scores']['positive']]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, scores, color=['#E57373', '#81C784'])
        
        # Add confidence value on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom')
        
        # Highlight the predicted sentiment
        predicted_idx = 0 if result['sentiment'] == 'negative' else 1
        bars[predicted_idx].set_color('gold')
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(2)
        
        # Set title and labels
        ax.set_title(f'Sentiment Analysis: {result["sentiment"].upper()} ({result["confidence"]:.4f})', fontsize=15)
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1.1)
        
        # Add the analyzed text as a subtitle
        text = result['text']
        if len(text) > 70:
            text = text[:67] + "..."
        plt.figtext(0.5, 0.01, f'"{text}"', wrap=True, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()

def main():
    analyzer = SentimentAnalyzer()
    
    # Get user input once
    print("\n=== Sentiment Analysis Tool ===")
    print("Enter text to analyze:")
    user_text = input("> ")
    
    if not user_text.strip():
        print("No text entered. Exiting.")
        return
    
    # Analyze the text
    print("Analyzing sentiment...")
    results = analyzer.analyze(user_text)
    result = results[0]  # Get the first (and only) result
    
    # Display textual results
    print("\nResults:")
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Negative score: {result['scores']['negative']:.4f}")
    print(f"Positive score: {result['scores']['positive']:.4f}")
    
    # Visualize the results
    analyzer.visualize_result(result)

if __name__ == "__main__":
    main()