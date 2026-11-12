# Model Evaluation Results

Roll No: 2025AG05429 | DM4ML Assignment 2

K = 10, relevance threshold = 4.0 stars, chronological train/test split.

| model               |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |   coverage |   diversity |   rmse |    mae |   n_users_ranked |   n_rating_pairs |
|:--------------------|------------------:|---------------:|-------------:|-----------:|------------:|-------:|-------:|-----------------:|-----------------:|
| popularity_baseline |            0.0888 |         0.0914 |       0.11   |     0.0297 |      0.7402 | 1.0397 | 0.8367 |               98 |             2949 |
| item_knn_cf         |            0.1112 |         0.0603 |       0.1279 |     0.0842 |      0.6957 | 1.0312 | 0.8001 |               98 |             2949 |

## Sample recommendations (top-5 shown)

|   user_id | model               | top_recommendations                                                                                                                                   |
|----------:|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
|        94 | popularity_baseline | Schindler's List (1993); Close Shave, A (1995); Wrong Trousers, The (1993); Star Wars (1977); Titanic (1997)                                          |
|        94 | item_knn_cf         | North by Northwest (1959); One Flew Over the Cuckoo's Nest (1975); Lawrence of Arabia (1962); To Kill a Mockingbird (1962); Rear Window (1954)        |
|       393 | popularity_baseline | Casablanca (1942); Close Shave, A (1995); Silence of the Lambs, The (1991); Rear Window (1954); Godfather, The (1972)                                 |
|       393 | item_knn_cf         | Raiders of the Lost Ark (1981); Terminator 2: Judgment Day (1991); Apollo 13 (1995); Terminator, The (1984); Aliens (1986)                            |
|       416 | popularity_baseline | Shawshank Redemption, The (1994); Casablanca (1942); Schindler's List (1993); Close Shave, A (1995); Wrong Trousers, The (1993)                       |
|       416 | item_knn_cf         | Raiders of the Lost Ark (1981); Star Wars (1977); Silence of the Lambs, The (1991); Shawshank Redemption, The (1994); Empire Strikes Back, The (1980) |
