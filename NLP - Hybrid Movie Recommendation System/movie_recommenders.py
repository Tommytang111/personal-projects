#Purpose: To recommend movies to users based on content and collaborative filtering.
#Model(s): 
#Tommy Tang
#Nov 17th, 2024

#LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, SVD, accuracy
import wget
import gc

#DATA SOURCE
"Datasets are already in the current directory."
#wget.download('https://files.grouplens.org/datasets/movielens/ml-20m.zip')
#wget.download('https://files.grouplens.org/datasets/movielens/ml-20m-README.html')

#IMPORT DATA (after unzipping)
tags=pd.read_csv('ml-20m/tags.csv')
movies=pd.read_csv('ml-20m/movies.csv')
ratings=pd.read_csv('ml-20m/ratings.csv')

#FUNCTIONS
def preprocess_data():
    """
    Preprocess data for content and collaborative filtering.
    """
    #Sort by movie ID for easier data manipulation
    tags.sort_values('movieId', inplace=True)

    # Collecting tags in movies dataframe for content
    for j in tags.movieId.unique():
        movies.loc[movies.loc[movies.movieId == j].index, 'tags'] = ' '.join(
            [i if type(i) == str else str(i) for i in tags.loc[(tags.movieId == j), 'tag'].unique().tolist()])

    # Splitting data for usability, larger datasets can crash
    use, dontuse = train_test_split(ratings, test_size=0.995)

    # Garbage collection to free up memory
    gc.collect()

    #Pivot table
    user_movies_data = pd.pivot_table(use, index='movieId', columns='userId', values='rating', fill_value=0)
    #Replacing NaN values with None
    movies['tags'] = movies['tags'].fillna('None')

    return use, user_movies_data

def content_filtering():
    """
    Content filtering using TF-IDF and SVD.
    """
    #Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer()

    #Fit and transform to same shape as movies dataframe
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=movies.index.tolist())

    #Initialize SVD
    svd = TruncatedSVD(n_components=19)
    #Fit and transform to latent matrix
    latent_matrix = svd.fit_transform(tfidf_df)

    # Justification for # of latent dimensions used
    """
    explained = svd.explained_variance_ratio_.cumsum()
    plt.plot(explained, '.-', ms = 16, color='red')
    plt.xlabel('Singular value components', fontsize= 12)
    plt.ylabel('Cumulative percent of variance', fontsize=12)        
    plt.show()
    """
    #Select 19 latent dimensions to dataframe
    latent_matrix_1_df = pd.DataFrame(latent_matrix[:, 0:19], index=movies['title'].tolist())
    return latent_matrix_1_df

def collaborative_filtering(user_movies_data, use):
    """
    Collaborative filtering using SVD.
    """
    svd = TruncatedSVD(n_components=20)
    latent_matrix_2 = svd.fit_transform(user_movies_data)

    latent_matrix_2_df = pd.DataFrame(latent_matrix_2, index=[movies.loc[(movies.movieId == i), 'title'].values[0] for i in (use['movieId'].unique())])
    return latent_matrix_2_df

def align_matrices(latent_matrix_1_df, latent_matrix_2_df):
    """
    Aligning the two latent matrices for hybrid filtering.
    """
    latent_matrix_1_df = latent_matrix_1_df.drop_duplicates()
    latent_matrix_2_df = latent_matrix_2_df.drop_duplicates()

    a = latent_matrix_2_df.copy()
    for i in latent_matrix_1_df.index:
        if i not in latent_matrix_2_df.index:
            a.loc[i] = np.zeros(20)

    b = latent_matrix_1_df.copy()
    for i in a.index:
        if i not in b.index:
            b.loc[i] = np.zeros(19)

    # Same index was repeated multiple times, decided to remove duplicates
    a = a[~a.index.duplicated(keep='first')]
    b = b[~b.index.duplicated(keep='first')]

    return a, b

def recommend_similar_movies(title, a, b):
    """"
    Return top 10 similar movies to the input movie title by hybrid score of content and collaborative filtering.
    """
    if title in b.index:
        a_1=np.array(b.loc[title]).reshape(1,-1)
        score_content=cosine_similarity(b,a_1).reshape(-1)
    else:
        score_content=0
    if title in a.index:
        a_2=np.array(a.loc[title]).reshape(1,-1)
        score_collab=cosine_similarity(a,a_2).reshape(-1)
    else:
        score_collab=0
    
    hybrid_score=(score_content+score_collab)/2

    dictDF={'content':score_content,'collab':score_collab,'hybrid':hybrid_score}
    dictDF
    similar_movies=pd.DataFrame(dictDF,index=a.index)


    similar_movies.sort_values('hybrid',ascending=False,inplace=True)
    return similar_movies[similar_movies['hybrid']>0].head(10)

def pred_user_rating(ui):
    """"
    Returns top 10 movie recommendations for the input user id using SVD.
    """
    if ui in ratings.userId.unique():
        ui_list = ratings[ratings.userId == ui].movieId.tolist()
        d = {k: v for k,v in Mapping_file.items() if not v in ui_list}        
        predictedL = []
        for i, j in d.items():     
            predicted = svd.predict(ui, j)
            predictedL.append((i, predicted[3])) 
        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])
        pdf.sort_values('ratings', ascending=False, inplace=True)  
        pdf.set_index('movies', inplace=True)    
        return pdf.head(10)        
    else:
        print("User Id does not exist in the list!")
        return None

def main():
    use, user_movies_data = preprocess_data()
    latent_matrix_1_df = content_filtering()
    latent_matrix_2_df = collaborative_filtering(user_movies_data, use)
    a, b = align_matrices(latent_matrix_1_df, latent_matrix_2_df)

    print('EXAMPLE RECOMMENDATIONS GIVEN MOVIE\n')
    print('Toy Story (1995):')
    print(recommend_similar_movies('Toy Story (1995)', a, b))
    print('\nMission Impossible II (2000):')
    print(recommend_similar_movies('Mission: Impossible II (2000)', a, b))

    global Mapping_file 
    Mapping_file = dict(zip(movies['title'].tolist(), movies['movieId'].tolist()))
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(use[['userId', 'movieId', 'rating']], reader)
    #Use different train_test_split than scikit-learn library
    from surprise.model_selection import train_test_split as tts
    # Using split data 'use' from earlier
    trainset, testset = tts(data, test_size=0.25)

    global svd 
    svd = SVD()
    svd.fit(trainset)
    preds = svd.test(testset)

    print('\nMODEL EVALUATION\n')
    accuracy.rmse(preds)
    accuracy.mae(preds)
    accuracy.fcp(preds)

    print('\nEXAMPLE RECOMMENDATIONS GIVEN USER:')
    print(pred_user_rating(100))
    print(pred_user_rating(1100))

    # Examine overall movie popularity by how mnay users have rated the movie
    ratings['movieId'].value_counts()
    counts = ratings['movieId'].value_counts().values.tolist()
    bestmovies = [movies.loc[(movies.movieId == i), 'title'].tolist()[0] for i in ratings['movieId'].value_counts().index.tolist()]
    best_movies_df = pd.DataFrame(counts, index=bestmovies).iloc[:20]
    print('\nTOP MOVIES BY # OF USERS RATED')
    best_movies_df.columns = ['Users']
    print(best_movies_df)

if __name__ == "__main__":
    main()