import gradio as gr
from fastai.vision.all import load_learner


learn = load_learner('pkg_recognizer_v1.pkl')

categories = learn.dls.vocab

def recognize_image(img):
    pred, idx, probs = learn.predict(img)
    
    return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()

examples = [
    '1.jpeg',
    '2.jpg',
    '3.jpg',
    '4.jpeg'
]

iface = gr.Interface(
    fn=recognize_image,
    inputs=image,
    outputs=label,
    examples=examples,
    title="Daily Object Packaging Image Classification",
    description="Classifies packaging into 12 categories: Bottled Water, Milk Cartons, Snack Chips, etc."
)

if __name__ == "__main__":
    iface.launch(inline=False)