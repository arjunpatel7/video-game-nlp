import gradio as gr
from datasets import load_dataset
from app.data_cleaning import prepare_document, cos_dicts, retrieve_top_k_similar
from sklearn.feature_extraction.text import TfidfVectorizer


# Create a big text interface allowing users to select the stuff we are interested in..


demo = gr.Blocks()

df = load_dataset("arjunpatel/best-selling-video-games")
df.set_format("pandas")
df = df["train"][:]

cleaned_wikis = df.wiki_page.apply(lambda x: prepare_document(x))
tfidf = TfidfVectorizer()
tfidf_wikis = tfidf.fit_transform(cleaned_wikis.tolist())
video_game_cos_dict = cos_dicts(df.Title, tfidf_wikis.toarray())


def find_similar_games(name, num):
    return retrieve_top_k_similar(name, video_game_cos_dict, num)

with demo:
    video_game = gr.Dropdown(df.Title.tolist(), default = df.Title.tolist()[0],
    label = "Selected Game")

    num_similar = gr.Dropdown([1, 2, 3, 4, 5], default = 1, label = "Number of Similar Games")

    find_similar = gr.Button("Find 'em!")

    output = gr.Textbox("Games will appear here!")

    find_similar.click(fn = find_similar_games, inputs = [video_game, num_similar],
    outputs = output)


demo.launch()
#drop down for video game

#drop down for number of similar games (1-5)

#button to retrieve