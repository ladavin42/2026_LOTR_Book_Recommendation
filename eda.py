import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Book Recommendation — Exploratory Data Analysis
    **Dataset:** Book-Crossing (Kaggle) — ~1.1M ratings, 270k books, 278k users

    We start by understanding the raw data before building any model.
    Three files: `Books.csv`, `Ratings.csv`, `Users.csv`
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    DATA = "data/"

    ratings = pd.read_csv(DATA + "Ratings.csv", sep=",", encoding="latin-1")
    books   = pd.read_csv(DATA + "Books.csv",   sep=",", encoding="latin-1",
                          on_bad_lines="skip")
    users   = pd.read_csv(DATA + "Users.csv",   sep=",", encoding="latin-1")

    # clean column names
    ratings.columns = ["user_id", "isbn", "rating"]
    books.columns   = ["isbn", "title", "author", "year", "publisher",
                       "img_s", "img_m", "img_l"]
    users.columns   = ["user_id", "location", "age"]

    # strip whitespace from string columns that will be used as keys
    ratings["isbn"] = ratings["isbn"].str.strip()
    books["isbn"]   = books["isbn"].str.strip()

    print(f"Ratings : {ratings.shape[0]:>10,} rows")
    print(f"Books   : {books.shape[0]:>10,} rows")
    print(f"Users   : {users.shape[0]:>10,} rows")
    return books, np, ratings


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Ratings: explicit vs implicit

    The rating scale is **0–10**, but 0 does not mean "bad".
    It means the user registered the book (read/owned it) but left no score.
    This is called **implicit feedback** — presence without opinion.

    Our first design decision: how do we treat 0s?
    """)
    return


@app.cell
def _(ratings):
    import plotly.express as px
    import plotly.graph_objects as go

    n_total    = len(ratings)
    n_implicit = (ratings["rating"] == 0).sum()
    n_explicit = (ratings["rating"]  > 0).sum()

    print(f"Total ratings  : {n_total:>10,}")
    print(f"Implicit (0)   : {n_implicit:>10,}  ({n_implicit/n_total*100:.1f}%)")
    print(f"Explicit (1-10): {n_explicit:>10,}  ({n_explicit/n_total*100:.1f}%)")

    # full distribution
    counts = ratings["rating"].value_counts().sort_index()
    fig = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Rating", "y": "Count"},
        title="Rating distribution (0 = implicit / unrated)",
        color=counts.index.astype(str),
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig.update_layout(showlegend=False)
    fig
    return n_explicit, n_implicit, n_total, px


@app.cell
def _(mo, n_explicit, n_implicit, n_total):
    mo.callout(
        mo.md(
            f"**{n_implicit/n_total*100:.0f}% of all rows are implicit (rating = 0).**  \n"
            "For our rating-based model we will drop these and work with the "
            f"{n_explicit:,} explicit ratings only.  \n"
            "We will also build a **Jaccard-based** model that uses *all* interactions "
            "(read vs not-read, ignoring score) so we can compare both signals."
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Explicit rating distribution & user bias

    Two things to check before any model:

    1. **Score skew** — do users tend to rate books they liked (positive bias)?
    2. **User bias** — some users always give 9s, others always give 4s.
       Pearson correlation handles this; raw averages do not.
    """)
    return


@app.cell
def _(px, ratings):
    explicit = ratings[ratings["rating"] > 0].copy()

    # per-user stats
    user_stats = (
        explicit.groupby("user_id")["rating"]
        .agg(n_rated="count", mean_rating="mean", std_rating="std")
        .reset_index()
    )

    print("Explicit ratings per user:")
    print(user_stats["n_rated"].describe().round(1).to_string())

    # distribution of per-user means  — shows individual bias
    fig2 = px.histogram(
        user_stats[user_stats["n_rated"] >= 5],   # at least 5 ratings
        x="mean_rating",
        nbins=40,
        title="Distribution of per-user average rating (users with ≥5 explicit ratings)",
        labels={"mean_rating": "User's average rating"},
        color_discrete_sequence=["#2a9d8f"],
    )
    fig2
    return (explicit,)


@app.cell
def _(mo):
    mo.md(r"""
    Most users rate books they chose to rate — so ratings skew high (7–9).
    This is called **positivity bias** and it is normal in recommendation datasets.

    **Why this matters for our model:** when computing user similarity we will
    centre each user's ratings around their own mean.
    That way a user whose scale is 6–8 and one whose scale is 8–10
    are treated consistently.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Sparsity — why collaborative filtering is hard
    """)
    return


@app.cell
def _(explicit):
    n_users_explicit = explicit["user_id"].nunique()
    n_books_explicit = explicit["isbn"].nunique()
    n_interactions   = len(explicit)

    matrix_size   = n_users_explicit * n_books_explicit
    sparsity      = 1 - n_interactions / matrix_size

    print(f"Users with ≥1 explicit rating : {n_users_explicit:>8,}")
    print(f"Books with ≥1 explicit rating : {n_books_explicit:>8,}")
    print(f"Explicit interactions         : {n_interactions:>8,}")
    print(f"Full matrix size              : {matrix_size:>8,}")
    print(f"Sparsity                      : {sparsity*100:.4f}%")
    return (sparsity,)


@app.cell
def _(mo, sparsity):
    mo.callout(
        mo.md(
            f"The rating matrix is **{sparsity*100:.2f}% empty**.  \n"
            "A typical user has rated fewer than 10 books out of 270,000.  \n"
            "This is why we cannot simply compute correlations on raw data — "
            "most pairs of users share *zero* books in common.  \n"
            "Our models must handle missing values carefully."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The long tail

    In almost every real-world dataset, a small number of books have
    thousands of ratings while the vast majority have only one or two.
    This is the **long tail** and it creates two problems:

    - Popular books dominate naive recommendations
    - Rare books cannot be recommended reliably (too little data)

    We will filter out very sparse users and books before modelling.
    """)
    return


@app.cell
def _(explicit, np, px):
    book_counts = explicit.groupby("isbn").size().sort_values(ascending=False)
    user_counts = explicit.groupby("user_id").size().sort_values(ascending=False)

    fig3 = px.histogram(
        x=np.log10(book_counts.values + 1),
        nbins=60,
        title="Book popularity — log10(number of explicit ratings)",
        labels={"x": "log₁₀(ratings)", "y": "Number of books"},
        color_discrete_sequence=["#e76f51"],
    )

    fig4 = px.histogram(
        x=np.log10(user_counts.values + 1),
        nbins=60,
        title="User activity — log10(number of explicit ratings given)",
        labels={"x": "log₁₀(ratings)", "y": "Number of users"},
        color_discrete_sequence=["#457b9d"],
    )

    pct_books_lt5 = (book_counts < 5).mean() * 100
    pct_users_lt5 = (user_counts < 5).mean() * 100

    print(f"Books with fewer than 5 explicit ratings : {pct_books_lt5:.1f}%")
    print(f"Users with fewer than 5 explicit ratings : {pct_users_lt5:.1f}%")

    fig3
    return (fig4,)


@app.cell
def _(fig4):
    fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Finding Lord of the Rings in the dataset

    The dataset uses ISBNs as book keys, not titles — so we need to find
    which ISBNs correspond to LOTR before we can do anything useful.
    """)
    return


@app.cell
def _(books, explicit):
    is_tolkien = books["author"].str.contains("Tolkien", case=False, na=False)
    is_lotr_title = (
        books["title"].str.contains("lord of the rings", case=False, na=False)
        | books["title"].str.contains("fellowship of the ring", case=False, na=False)
        | books["title"].str.contains("two towers", case=False, na=False)
        | books["title"].str.contains("return of the king", case=False, na=False)
        | books["title"].str.contains("hobbit", case=False, na=False)
    )
    # must be by Tolkien — excludes discussion/companion books by other authors
    lotr_mask = is_tolkien & is_lotr_title

    lotr_books = books[lotr_mask][["isbn", "title", "author", "year"]].copy()

    # count how many explicit ratings each LOTR edition has
    lotr_ratings_count = (
        explicit[explicit["isbn"].isin(lotr_books["isbn"])]
        .groupby("isbn")
        .size()
        .rename("n_ratings")
        .reset_index()
    )
    lotr_books = lotr_books.merge(lotr_ratings_count, on="isbn", how="left")
    lotr_books["n_ratings"] = lotr_books["n_ratings"].fillna(0).astype(int)
    lotr_books = lotr_books.sort_values("n_ratings", ascending=False)

    print(f"LOTR-related entries in Books.csv: {len(lotr_books)}")
    print(f"Total explicit ratings across all editions: {lotr_books['n_ratings'].sum()}")
    lotr_books.head(20)
    return (lotr_books,)


@app.cell
def _(explicit, lotr_books):
    lotr_isbns = lotr_books["isbn"].tolist()
    lotr_explicit = explicit[explicit["isbn"].isin(lotr_isbns)].copy()

    lotr_raters = lotr_explicit["user_id"].unique()

    print(f"Unique ISBNs  : {len(lotr_isbns)}")
    print(f"Unique users who rated ≥1 LOTR book: {len(lotr_raters)}")
    print()
    print("Rating distribution for LOTR books:")
    print(lotr_explicit["rating"].value_counts().sort_index().to_string())
    return lotr_explicit, lotr_isbns, lotr_raters


@app.cell
def _(lotr_explicit, px):
    fig5 = px.histogram(
        lotr_explicit, x="rating", nbins=10,
        title="LOTR rating distribution (explicit only)",
        labels={"rating": "Rating (1–10)"},
        color_discrete_sequence=["#6a4c93"],
    )
    fig5
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. What do LOTR readers also read?

    This is the core intuition behind our recommendation approach.
    Users who rated LOTR — what other books did they rate highly?

    This is the simplest possible "model": a **popularity baseline among LOTR readers**.
    It sets the bar that our collaborative filtering model needs to beat.
    """)
    return


@app.cell
def _(books, explicit, lotr_isbns, lotr_raters):
    # all explicit ratings by users who rated LOTR, excluding LOTR itself
    lotr_reader_ratings = explicit[
        explicit["user_id"].isin(lotr_raters)
        & ~explicit["isbn"].isin(lotr_isbns)
    ].copy()

    # join to get titles
    lotr_reader_ratings = lotr_reader_ratings.merge(
        books[["isbn", "title", "author"]], on="isbn", how="left"
    )

    # aggregate: count of raters + mean rating
    baseline = (
        lotr_reader_ratings.groupby(["isbn", "title", "author"])
        .agg(
            n_raters=("user_id", "nunique"),
            mean_rating=("rating", "mean"),
        )
        .reset_index()
        .sort_values("n_raters", ascending=False)
    )

    print("Top 20 books by number of LOTR readers who rated them:\n")
    print(
        baseline.head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary & next steps

    **What we learned from EDA:**

    - ~72% of ratings are implicit (0) — we will separate the two signals
    - Users skew positive — we must centre ratings per user before computing similarity
    - The matrix is >99.99% sparse — we need filtering thresholds
    - LOTR has multiple ISBNs (different editions) — we group them all
    - The popularity baseline already produces recognisable recommendations

    **What we build next:**

    1. **Data cleaning** — apply sparsity filters, decide 0-rating strategy
    2. **User-based CF** — Pearson similarity between users
    3. **Item-based CF** — book-book similarity matrix
    4. **SVD** — matrix factorisation for latent factor recommendations
    5. **Open Library enrichment** — fetch genres/subjects per ISBN
    6. **Comparison** — do the models agree? where do they differ?
    """)
    return


if __name__ == "__main__":
    app.run()
