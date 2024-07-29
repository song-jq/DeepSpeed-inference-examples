from transformers  import pipeline

nlp = pipeline("fill-mask", model="./bert-base")

text = "The quick brown [MASK] jumps over the lazy dog."

print(nlp(text))