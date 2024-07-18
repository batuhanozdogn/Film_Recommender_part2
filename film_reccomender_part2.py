import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


data = pd.read_csv("/Users/batuhanozdogan/Downloads/movies_metadata.csv", low_memory=False)
veri = data.copy()

df = veri[["id", "title", "overview"]]
df.loc[:, "overview"] = df["overview"].fillna(" ")

tdıdf = TfidfVectorizer(stop_words="english")
tdıdf_metric = tdıdf.fit_transform(df["overview"])

assimilart_matrix = linear_kernel(tdıdf_metric, tdıdf_metric)

indexs = pd.Series(df.index, index=df["title"]).drop_duplicates()
film_index = indexs["Toy Story"]

film_simitray = list(enumerate(assimilart_matrix[film_index]))
Lıst = sorted(film_simitray, key=lambda x: x[1], reverse=True)
Lıst_point = Lıst[1:6]  yoruz
Lıst_index = [i[0] for i in Lıst_point]
result = df["title"].iloc[Lıst_index]
print(result)
