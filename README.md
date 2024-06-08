# ytsum

## Summarize YouTube videos instantly with the power of distill-whisper and Mixtral-8x7B

ytsum is a Python package that allows you to summarize YouTube videos by leveraging the power of distil-whisper for transcription and Mixtral-8x7B for generating detailed answers based on the video content.

## Setup

To use ytsum, you need to get an API key from Together AI. 

You can sign up and obtain your API key here: [Together AI](https://www.together.ai/).

## Installation

```bash
pip install ytsum
```
```python
from ytsum import answer_youtube_question, set_together_api_key

# Set your Together AI API key
set_together_api_key("your_together_ai_api_key")

# Example usage
youtube_url = "https://www.youtube.com/watch?v=example"
query = "What is the main topic of this video?"

result = answer_youtube_question(youtube_url, query)

print(result)
```
## Models used

Distil-large-v3: https://huggingface.co/distil-whisper/distil-large-v3

Mixtral-8x7B-SFT: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT
