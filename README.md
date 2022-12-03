# 🎮 Video Games NLP

An app to conduct semantic search over best selling Nintendo Switch and PS4 video games!
Launch the gradio app to learn about TF-IDF based comparison search and natural language search
using Sentence Transformers!

Want to try it for yourself? Follow the instructions 
below or [click here](https://huggingface.co/spaces/arjunpatel/best-selling-video-games) for the deployed version!

## Motivation

I made this app for a talk I did with a data science group called Zindi. I gave a small lecture on Word and Document Embeddings, 
scraped a dataset from Wikipedia on video games and their respective wiki pages, and built a Gradio tool to help students explore embedding tech.

I wanted to showcase the power of TF-IDF embedding search (which is one tab of the demo), and also search that uses any query, not limited to the game corpus. 
The second tab allows you to type any request, and the model will aim to find a semantically similar entry in its database. Pretty cool! 
## Requirements
* gradio
* sentence transformers
* datasets (from huggingface) (for image generation)
* pytorch
* scikit-learn

## Installation

Clone this repo, and run the following command:
```bash
gradio app/app.py
```
The app will open in a local browser!

## Author
This app was made by Arjun Patel, a data scientist with experience
applying deep learning to audio and text data. 
Connect with me on [Linkedin](https://www.linkedin.com/in/arjunkirtipatel/)!

