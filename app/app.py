import gradio as gr
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from app.data_cleaning import prepare_document, cos_dicts, retrieve_top_k_similar
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


demo = gr.Blocks()

df = load_dataset("arjunpatel/best-selling-video-games")
df.set_format("pandas")
df = df["train"][:]

cleaned_wikis = df.wiki_page.apply(lambda x: prepare_document(x))
tfidf = TfidfVectorizer()
tfidf_wikis = tfidf.fit_transform(cleaned_wikis.tolist())
video_game_cos_dict = cos_dicts(df.Title, tfidf_wikis.toarray())

embedder = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')
msmarco_embeddings = embedder.encode(df.wiki_page.tolist(), convert_to_tensor = True)

def nli_search(query):
    # given a query, return top few similar games

    # example code taken from Sentence Transformers docs
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, msmarco_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    #print("\n\n======================\n\n")
    #print("Query:", query)
    #print("\nTop 5 most similar sentences in corpus:")
    ret_list = []

    for score, idx in zip(top_results[0], top_results[1]):
        ret_list.append((df.wiki_page.tolist()[idx][0:100], "(Score: {:.4f})".format(score)))
    
    return ret_list


def find_similar_games(name, num):
    return retrieve_top_k_similar(name, video_game_cos_dict, num)

with demo:
    gr.Markdown("<h1><center>Find your next Video Game!</center></h1>")
    gr.Markdown(
        """This Gradio demo allows you to search a list of best selling video games and their corresponding Wikipedia pages
        using NLP! The first tab allows for a TF-IDF based search, and the second leverages Sentence Transformers for a Natural Language
        Search. Enjoy!""")
    with gr.Tab("TF-IDF Similarity Search"):
        video_game = gr.Dropdown(df.Title.tolist(), default = df.Title.tolist()[0],
        label = "Selected Game")

        num_similar = gr.Dropdown([1, 2, 3, 4, 5], default = 1, label = "Number of Similar Games")

        find_similar = gr.Button("Find 'em!")

        output = gr.Textbox("Games will appear here!")

        find_similar.click(fn = find_similar_games, inputs = [video_game, num_similar],
        outputs = output)

    with gr.Tab("Natural Language Search"):
        q = gr.Textbox("Type a query here. Try: find me mario games")
        find_nli = gr.Button("Search!")
        nli_output = gr.Textbox("Output will appear here from NLI search")

        find_nli.click(fn = nli_search, inputs = [q], outputs = nli_output)
    


demo.launch()
#drop down for video game

#drop down for number of similar games (1-5)

#button to retrieve