import gradio as gr
from gliner import GLiNER
import hashlib
import os
import gdown
import zipfile
import colorsys

_models = {}

AVAILABLE_MODELS = {
    "GLiNER-S": "urchade/gliner_small-v2.1",
    "GLiNER-M": "urchade/gliner_medium-v2.1",
    "GLiNER-L": "urchade/gliner_large-v2.1",
    "GliNER-Multi": "urchade/gliner_multi-v2.1",
    "GLiNER-S-Team10": "./models/GLiNER-S-Team10",
    "GLiNER-M-Team10": "./models/GLiNER-M-Team10",
    "GLiNER-Multi-Team10": "./models/GLiNER-Multi-Team10",
}

MODEL_KEYS = list(AVAILABLE_MODELS.keys())

GDRIVE_MODEL_LINKS = {
    "GLiNER-S-Team10": "https://drive.google.com/file/d/1cx48Fkwkhj8iXiOHeyNGhClL2eHDPbIS/view?usp=sharing",
    "GLiNER-M-Team10": "https://drive.google.com/file/d/1d7PB3mBWDfUAcczpW8QJsO-uh5BSOxv4/view?usp=sharing",
    "GLiNER-Multi-Team10": "https://drive.google.com/file/d/1a60iv-fUCyvZs9bGCH2evtL_swaDAdXn/view?usp=sharing",
}

EXAMPLES = [
    [
        """Barack Obama was born in Honolulu, Hawaii, in 1961, and spent part of his early childhood
living abroad with his family before returning to the mainland United States.

After graduating from Columbia University and Harvard Law School, Obama began his political
career in Chicago, where he worked as a community organizer and later taught constitutional law.

He rose to national prominence after delivering a keynote address at the Democratic National
Convention, eventually becoming the 44th President of the United States from 2009 to 2017.

During his presidency, Obama represented the United States at multiple United Nations General
Assembly meetings in New York and engaged in diplomatic talks with leaders from Europe and Asia.

Following his departure from the White House, Obama founded the Obama Foundation, an organization
focused on leadership development, civic engagement, and global youth empowerment initiatives.""",
        "Person, Location, Country, Organization, Political Position, University, Date",
    ],
    [
        """Apple Inc. officially unveiled the iPhone 15 Pro during a large-scale keynote event held
at Apple Park in Cupertino, California, drawing millions of viewers from around the world.

The presentation was led by CEO Tim Cook, who emphasized advancements in chip design, photography,
and sustainability, while the event was streamed live on platforms such as YouTube and Apple TV.

Technology journalists from Bloomberg, The New York Times, and Reuters published in-depth analyses
of the product launch, highlighting its potential impact on the global smartphone market.

Following the announcement, Apple shares rose significantly on the NASDAQ stock exchange, as
investors reacted positively to early sales forecasts and analyst reports.""",
        "Organization, Product, Person, Location, Event, Platform, Media, Stock Exchange",
    ],
    [
        """The Michelin-starred restaurant Noma, located in Copenhagen, Denmark, has become widely
recognized for redefining modern Nordic cuisine through experimentation and seasonal ingredients.

Founded by chef René Redzepi, the restaurant emphasizes fermentation, foraging, and close
collaboration with local producers, setting new standards in fine dining worldwide.

Noma has been ranked multiple times as the best restaurant in the world by The World’s 50 Best
Restaurants organization, influencing chefs across Europe, Asia, and North America.

Beyond its culinary achievements, the restaurant has also served as a research lab, publishing
experiments and hosting international chefs interested in innovative gastronomy.""",
        "Restaurant, Location, Country, Cuisine, Person, Award, Organization",
    ],
    [
        """The science fiction film Interstellar, directed by Christopher Nolan, was released in 2014
and quickly became a global box office success, earning both critical acclaim and commercial success.

The movie starred Matthew McConaughey as a former NASA pilot tasked with leading a mission beyond
the Milky Way galaxy in search of a new habitable planet for humanity.

Anne Hathaway, Jessica Chastain, and Michael Caine played key supporting roles, contributing to
the film’s emotional depth and narrative complexity.

The musical score, composed by Hans Zimmer, was widely praised for its innovative use of organ
music and became closely associated with the film’s identity.""",
        "Movie, Director, Actor, Composer, Date, Genre, Organization",
    ],
    [
        """Taylor Swift released her album Midnights in October 2022, marking a stylistic shift toward
introspective songwriting inspired by personal experiences and late-night reflections.

The album broke multiple streaming records on platforms such as Spotify and Apple Music, becoming
one of the most streamed releases within its first 24 hours.

Songs like Anti-Hero, Bejeweled, and Lavender Haze dominated global music charts and received
extensive radio airplay across North America and Europe.

Midnights went on to win several awards at major music ceremonies, including the Grammy Awards,
further solidifying Swift’s influence in the global music industry.""",
        "Musical Artist, Album, Song, Date, Streaming Platform, Award, Location",
    ],
    [
        """In March 2024, researchers at Google DeepMind published a breakthrough paper on large-scale
reinforcement learning in the prestigious scientific journal Nature.

The study proposed novel neural architectures capable of learning complex strategies with
significantly reduced computational requirements compared to previous approaches.

The research was conducted in collaboration with scientists from the University of Oxford and
the Massachusetts Institute of Technology, highlighting cross-institutional cooperation.

Experts believe the findings could have long-term implications for robotics, healthcare,
and autonomous systems development.""",
        "Organization, Date, Research Field, Journal, University",
    ],
    [
        """The UEFA Champions League final was held at Wembley Stadium in London, attracting tens of
thousands of spectators and millions of television viewers worldwide.

Real Madrid faced Borussia Dortmund in a highly anticipated match that featured dramatic goals,
tactical adjustments, and intense competition throughout the game.

After securing victory, Real Madrid extended its legacy as one of the most successful clubs in
European football history, while Dortmund received praise for its performance.

The event was organized by UEFA and generated substantial revenue through broadcasting rights,
sponsorship deals, and international media coverage.""",
        "Sports Event, Stadium, Location, SportsTeam, Organization",
    ],
    [
        """Nguyễn Nhật Ánh là một nhà văn nổi tiếng tại Việt Nam, được nhiều thế hệ độc giả yêu mến
nhờ phong cách viết giản dị, gần gũi và giàu cảm xúc.

Ông được biết đến với hàng loạt tác phẩm văn học dành cho thanh thiếu niên như “Mắt biếc”,
“Cho tôi xin một vé đi tuổi thơ”, và nhiều truyện dài khác.

Các tác phẩm của ông thường lấy bối cảnh đời sống học đường và tuổi thơ tại Việt Nam, phản ánh
những giá trị nhân văn sâu sắc.

Nhiều tác phẩm của Nguyễn Nhật Ánh đã được chuyển thể thành phim điện ảnh và công chiếu tại
Hà Nội cũng như Thành phố Hồ Chí Minh, thu hút đông đảo khán giả.""",
        "Person, Country, Literary Work, Movie, Location",
    ],
]


def ensure_model_exists(model_path: str):
    if not model_path.startswith("./models/"):
        return

    if os.path.exists(model_path):
        return

    model_name = os.path.basename(model_path)
    if model_name not in GDRIVE_MODEL_LINKS:
        raise RuntimeError(f"No Google Drive link for model: {model_name}")

    os.makedirs("./models", exist_ok=True)
    zip_path = f"./models/{model_name}.zip"

    gdown.download(GDRIVE_MODEL_LINKS[model_name], zip_path, quiet=False, fuzzy=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("./models")

    os.remove(zip_path)


def get_model(model_name: str):
    if model_name not in _models:
        ensure_model_exists(model_name)
        _models[model_name] = GLiNER.from_pretrained(model_name)
    return _models[model_name]


def label_to_color(label: str) -> str:
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)

    hue = (h % 360) / 360.0
    saturation = 0.55
    lightness = 0.55

    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255),
        int(g * 255),
        int(b * 255),
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


ENTITY_CSS = """
.ner-text-box {
  background: var(--block-background-fill);
  border: 1px solid var(--block-border-color);
  border-radius: 8px;
  padding: 16px;
  line-height: 1.7;
  margin-bottom: 16px;
}

.ner-text-box p {
  margin: 0 0 12px 0;
}

.model-title {
  font-weight: 600;
  margin-bottom: 8px;
}

.ent {
  display: inline-flex;
  align-items: center;
  padding: 2px;
  padding-left: 6px;
  margin: 2px 4px;
  border-radius: 4px;
  font-weight: 500;
}

.ent-text {
  line-height: 1;
}

.ent .tag {
  margin-left: 6px;
  padding: 2px 6px;
  border-radius: 2px;
  font-size: 0.75em;
  line-height: 1;
  font-weight: bold;
  color: white;
}
"""


def preserve_linebreaks(text: str) -> str:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    html_paragraphs = []

    for p in paragraphs:
        html_paragraphs.append(f"<p>{p}</p>")

    return "".join(html_paragraphs)


def rewrite_text_with_entities(text, entities):
    rewritten = text
    entities = sorted(entities, key=lambda x: x["start"], reverse=True)

    for e in entities:
        span = text[e["start"] : e["end"]]
        tag_color = label_to_color(e["label"])
        bg_color = hex_to_rgba(tag_color, alpha=0.4)

        replacement = (
            f"<span class='ent' style='background:{bg_color}'>"
            f"<span class='ent-text'>{span}</span>"
            f"<span class='tag' style='background:{tag_color}'>"
            f"{e['label']} ({e['score']:.2f})"
            f"</span></span>"
        )

        rewritten = rewritten[: e["start"]] + replacement + rewritten[e["end"] :]

    return preserve_linebreaks(rewritten)


def ner_inference(text, tag_string, selected_models):
    labels = [t.strip() for t in tag_string.split(",") if t.strip()]

    html_blocks = []
    tables = []
    accordions_visibility = []

    for model_key in MODEL_KEYS:
        if model_key not in selected_models:
            tables.append([])
            accordions_visibility.append(gr.update(visible=False))
            continue

        model = get_model(AVAILABLE_MODELS[model_key])
        entities = model.predict_entities(text, labels)

        html_blocks.append(
            f"<div class='ner-text-box'>"
            f"<div class='model-title'>{model_key}</div>"
            f"{rewrite_text_with_entities(text, entities)}"
            f"</div>"
        )

        rows = []
        for i, e in enumerate(entities, 1):
            rows.append(
                [
                    i,
                    text[e["start"] : e["end"]],
                    e["label"],
                    e["start"],
                    e["end"],
                    round(e["score"], 4),
                ]
            )

        tables.append(rows)
        accordions_visibility.append(gr.update(visible=True))

    return ["".join(html_blocks)] + tables + accordions_visibility


with gr.Blocks(title="NER Playground") as demo:
    gr.Markdown("# NER Playground")

    model_select = gr.Dropdown(
        choices=MODEL_KEYS,
        value=["GLiNER-S-Team10", "GLiNER-M-Team10"],
        multiselect=True,
        label="Models",
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Input text",
            placeholder="Enter text for NER here...",
            lines=6,
        )
        tag_input = gr.Textbox(
            label="Entity tags",
            placeholder="Enter comma-separated entity tags (e.g., Person, Organization, Location)...",
        )

    with gr.Accordion("Examples (click to autofill)", open=False):
        gr.Examples(examples=EXAMPLES, inputs=[text_input, tag_input])

    run_btn = gr.Button("Run NER", variant="primary")

    gr.Markdown("## Output")
    rewritten_output = gr.HTML()

    model_tables = []
    model_accordions = []

    gr.Markdown("### Extracted Entities")

    for model_key in MODEL_KEYS:
        accordion = gr.Accordion(model_key, open=False, visible=False)
        with accordion:
            table = gr.Dataframe(
                headers=["Index", "Text", "Label", "Start", "End", "Score"],
                interactive=False,
            )
        model_accordions.append(accordion)
        model_tables.append(table)

    run_btn.click(
        ner_inference,
        inputs=[text_input, tag_input, model_select],
        outputs=[rewritten_output] + model_tables + model_accordions,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio Hub",
        default=False,
    )
    args = parser.parse_args()

    demo.launch(css=ENTITY_CSS, share=args.share)
