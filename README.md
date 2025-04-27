# Sentiment Analysis Tool

A Python tool for analyzing sentiment in text using the Hugging Face Transformers library and pre-trained DistilBERT model.

## Overview

This tool analyzes the sentiment of input text and classifies it as positive or negative. It provides confidence scores and visualizes the results using matplotlib.

## Features

- Text sentiment classification (positive/negative)
- Confidence score calculation
- Visual representation of sentiment analysis results
- Support for GPU acceleration (when available)
- Simple command-line interface

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- Matplotlib
- NumPy

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:

```bash
pip install torch transformers matplotlib numpy
```

## Usage

Run the script:

```bash
python sentiment_analyzer.py
```

Enter your text when prompted, and the tool will:
1. Analyze the sentiment
2. Display detailed results in the terminal
3. Show a visualization of the confidence scores

## Example Output

```
=== Sentiment Analysis Tool ===
Enter text to analyze:
> I really enjoyed this movie! The actors were amazing and the plot was engaging.

Analyzing sentiment...

Results:
Text: I really enjoyed this movie! The actors were amazing and the plot was engaging.
Sentiment: POSITIVE
Confidence: 0.9964
Negative score: 0.0036
Positive score: 0.9964
```

The tool will also display a bar chart visualization showing the confidence scores for both positive and negative sentiment, with the predicted sentiment highlighted.

## Model Information

This tool uses the `distilbert-base-uncased-finetuned-sst-2-english` model by default. This model is a fine-tuned version of DistilBERT trained on the Stanford Sentiment Treebank (SST-2) dataset, which consists of movie reviews labeled with binary sentiment (positive/negative).

## Customization

You can modify the code to:
- Use a different pre-trained model
- Adjust visualization parameters
- Process batch inputs from files
- Fine-tune the model on your own dataset

## Advanced Usage

### Analyzing Multiple Texts

You can modify the code to analyze multiple texts by passing a list to the `analyze` method:

```python
analyzer = SentimentAnalyzer()
texts = [
    "I love this product so much!",
    "The service was terrible and I would not recommend it.",
    "It was okay, nothing special."
]
results = analyzer.analyze(texts)
```

### Using a Different Model

You can use a different pre-trained model by specifying it in the constructor:

```python
analyzer = SentimentAnalyzer(model_name="roberta-base-go-emotion")
```

## License

This project is open source and available under the MIT License.

## Credits

This tool uses pre-trained models from the Hugging Face Transformers library, which provides state-of-the-art NLP models and tools.