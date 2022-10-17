import translators as ts

import pandas as pd

df = pd.DataFrame(corpus, columns = ['text'])
df['welsh_text'] = df['text'].apply(lambda x: ts.google(x, from_language='en', to_language='cy'))