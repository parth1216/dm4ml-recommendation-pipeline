# Data Validation Report

Roll No: 2025AG05429 | DM4ML Assignment 2

Executed (UTC): 2026-11-12T20:47:16Z

## Entity summary

| Entity | Rows in | Valid | Quarantined | Error file |
|---|---:|---:|---:|---|
| users | 943 | 943 | 0 | `data/errors/users_errors.csv` |
| items | 1682 | 1682 | 0 | `data/errors/items_errors.csv` |
| interactions | 100000 | 100000 | 0 | `data/errors/interactions_errors.csv` |

## Checks executed

| Entity | Check | Status | Failed rows |
|---|---|---|---:|
| users | no missing values in required columns | PASS | 0 |
| users | user_id: greater_than_or_equal_to(1) | PASS | 0 |
| users | age: in_range(5, 100) | PASS | 0 |
| users | gender: isin(['M', 'F']) | PASS | 0 |
| items | no missing values in required columns | PASS | 0 |
| items | movie_id: greater_than_or_equal_to(1) | PASS | 0 |
| items | unknown: isin([0, 1]) | PASS | 0 |
| items | Action: isin([0, 1]) | PASS | 0 |
| items | Adventure: isin([0, 1]) | PASS | 0 |
| items | Animation: isin([0, 1]) | PASS | 0 |
| items | Children: isin([0, 1]) | PASS | 0 |
| items | Comedy: isin([0, 1]) | PASS | 0 |
| items | Crime: isin([0, 1]) | PASS | 0 |
| items | Documentary: isin([0, 1]) | PASS | 0 |
| items | Drama: isin([0, 1]) | PASS | 0 |
| items | Fantasy: isin([0, 1]) | PASS | 0 |
| items | Film-Noir: isin([0, 1]) | PASS | 0 |
| items | Horror: isin([0, 1]) | PASS | 0 |
| items | Musical: isin([0, 1]) | PASS | 0 |
| items | Mystery: isin([0, 1]) | PASS | 0 |
| items | Romance: isin([0, 1]) | PASS | 0 |
| items | Sci-Fi: isin([0, 1]) | PASS | 0 |
| items | Thriller: isin([0, 1]) | PASS | 0 |
| items | War: isin([0, 1]) | PASS | 0 |
| items | Western: isin([0, 1]) | PASS | 0 |
| interactions | no missing values in required columns | PASS | 0 |
| interactions | user_id: greater_than_or_equal_to(1) | PASS | 0 |
| interactions | movie_id: greater_than_or_equal_to(1) | PASS | 0 |
| interactions | rating: in_range(1.0, 5.0) | PASS | 0 |
| interactions | timestamp: in_range(788918400, 1794516436) | PASS | 0 |
| interactions | dataframe: duplicate (user_id, movie_id, timestamp) event | PASS | 0 |
| interactions | referential integrity (user_id in users, movie_id in items) | PASS | 0 |
