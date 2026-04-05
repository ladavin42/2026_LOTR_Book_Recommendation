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
    # Book Recommendation — "I like Lord of the Rings, what else should I read?"

    **Dataset:** Book-Crossing (Kaggle) — ~1.1M ratings, 270k books, 278k users

    This notebook is in two parts:
    - **Part 1 — EDA**: understand the raw data before building anything
    - **Part 2 — Models**: four progressively more sophisticated recommenders
    """)
    return


# ── IMPORTS ───────────────────────────────────────────────────────────────────

@app.cell
def imports():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    import time
    from scipy.sparse.linalg import svds
    return go, np, pd, px, requests, svds, time


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

@app.cell
def constants():
    DATA             = "data/"
    MIN_USER_RATINGS = 20    # drop users with fewer explicit ratings
    MIN_BOOK_RATINGS = 20    # drop books with fewer explicit ratings
    K_LATENT         = 50    # SVD latent dimensions
    TOP_N            = 15    # recommendations to return per model
    return DATA, K_LATENT, MIN_BOOK_RATINGS, MIN_USER_RATINGS, TOP_N


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@app.cell
def load_data(pd, DATA):
    ratings = pd.read_csv(DATA + "Ratings.csv", encoding="latin-1")
    books   = pd.read_csv(DATA + "Books.csv",   encoding="latin-1", on_bad_lines="skip")
    users   = pd.read_csv(DATA + "Users.csv",   encoding="latin-1")

    ratings.columns = ["user_id", "isbn", "rating"]
    books.columns   = ["isbn", "title", "author", "year", "publisher",
                       "img_s", "img_m", "img_l"]
    users.columns   = ["user_id", "location", "age"]

    ratings["isbn"] = ratings["isbn"].str.strip()
    books["isbn"]   = books["isbn"].str.strip()

    explicit = ratings[ratings["rating"] > 0].copy()

    print(f"Ratings : {len(ratings):>10,} rows  (including 0-rated)")
    print(f"Books   : {len(books):>10,} rows")
    print(f"Users   : {len(users):>10,} rows")
    print(f"Explicit: {len(explicit):>10,} rows  (rating 1–10)")
    return books, explicit, ratings


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("## 1 · Ratings: explicit vs implicit")
    return


@app.cell
def _(ratings, px):
    _n_total    = len(ratings)
    _n_implicit = (ratings["rating"] == 0).sum()
    _n_explicit = (ratings["rating"]  > 0).sum()

    print(f"Total ratings  : {_n_total:>10,}")
    print(f"Implicit (0)   : {_n_implicit:>10,}  ({_n_implicit/_n_total*100:.1f}%)")
    print(f"Explicit (1-10): {_n_explicit:>10,}  ({_n_explicit/_n_total*100:.1f}%)")

    _counts = ratings["rating"].value_counts().sort_index()
    _fig = px.bar(
        x=_counts.index, y=_counts.values,
        labels={"x": "Rating", "y": "Count"},
        title="Rating distribution  (0 = implicit / read but not scored)",
        color=_counts.index.astype(str),
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    _fig.update_layout(showlegend=False)
    _fig


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**62% of all rows are implicit (rating = 0).**  \n"
            "These are books users registered but never scored — not bad ratings.  \n"
            "For rating-based models we drop them. "
            "A Jaccard model (read/not-read, ignoring score) could use them as a weak signal."
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Explicit rating distribution & user bias

    Two things to check before modelling:

    1. **Score skew** — do users only rate books they liked?
    2. **User bias** — some users give everything 9, others give everything 5.
       Pearson correlation handles this; raw averages do not.
    """)
    return


@app.cell
def _(explicit, px):
    _user_stats = (
        explicit.groupby("user_id")["rating"]
        .agg(n_rated="count", mean_rating="mean")
        .reset_index()
    )

    print("Explicit ratings per user:")
    print(_user_stats["n_rated"].describe().round(1).to_string())

    _fig = px.histogram(
        _user_stats[_user_stats["n_rated"] >= 5],
        x="mean_rating", nbins=40,
        title="Per-user average rating  (users with ≥5 explicit ratings)",
        labels={"mean_rating": "User's average rating"},
        color_discrete_sequence=["#2a9d8f"],
    )
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    Ratings skew high (7–9) — users mostly rate books they chose to read and enjoyed.
    This is called **positivity bias** and is normal in recommendation datasets.

    This is why we mean-centre each user's ratings before computing similarity:
    a user whose scale is 6–8 and one whose scale is 8–10
    are treated consistently after centring.
    """)
    return


@app.cell
def _(mo):
    mo.md("## 3 · Sparsity — the core challenge")
    return


@app.cell
def _(explicit):
    _n_users = explicit["user_id"].nunique()
    _n_books = explicit["isbn"].nunique()
    _n_inter = len(explicit)
    _size    = _n_users * _n_books
    sparsity = 1 - _n_inter / _size

    print(f"Users with ≥1 explicit rating : {_n_users:>8,}")
    print(f"Books with ≥1 explicit rating : {_n_books:>8,}")
    print(f"Explicit interactions         : {_n_inter:>8,}")
    print(f"Full matrix size              : {_size:>8,}")
    print(f"Sparsity                      : {sparsity*100:.3f}%")
    return (sparsity,)


@app.cell
def _(mo, sparsity):
    mo.callout(
        mo.md(
            f"The rating matrix is **{sparsity*100:.2f}% empty**.  \n"
            "A typical user has rated fewer than 10 books out of 270,000.  \n"
            "Most pairs of users share *zero* books in common — "
            "our models must handle missing values carefully."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The long tail

    A small number of books have thousands of ratings; the vast majority have one or two.
    This **power-law distribution** creates two problems:

    - Popular books dominate naive recommendations
    - Rare books cannot be recommended reliably (too little data)

    We log the x-axis so the shape is visible — on a linear scale
    everything would be crammed into the leftmost bar.
    """)
    return


@app.cell
def _(explicit, np, px):
    _book_counts = explicit.groupby("isbn").size().sort_values(ascending=False)
    _user_counts = explicit.groupby("user_id").size().sort_values(ascending=False)

    print(f"Books with fewer than 5 explicit ratings : {(_book_counts < 5).mean()*100:.1f}%")
    print(f"Users with fewer than 5 explicit ratings : {(_user_counts < 5).mean()*100:.1f}%")

    _fig_books = px.histogram(
        x=np.log10(_book_counts.values + 1), nbins=60,
        title="Book popularity — log₁₀(number of explicit ratings)",
        labels={"x": "log₁₀(ratings)", "y": "Number of books"},
        color_discrete_sequence=["#e76f51"],
    )
    _fig_books


@app.cell
def _(explicit, np, px):
    _user_counts = explicit.groupby("user_id").size().sort_values(ascending=False)
    _fig_users = px.histogram(
        x=np.log10(_user_counts.values + 1), nbins=60,
        title="User activity — log₁₀(number of explicit ratings given)",
        labels={"x": "log₁₀(ratings)", "y": "Number of users"},
        color_discrete_sequence=["#457b9d"],
    )
    _fig_users


# ── LOTR IDENTIFICATION ───────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Finding Lord of the Rings in the dataset

    The dataset keys on ISBN, not title — so we need to find every ISBN that
    corresponds to a Tolkien book before we can do anything useful.

    We require the author to contain **"Tolkien"** AND **"J." or "John"**
    to exclude Christopher Tolkien's editorial/companion works.
    """)
    return


@app.cell
def identify_lotr(books):
    _is_tolkien = (
        books["author"].str.contains("Tolkien", case=False, na=False)
        & (
            books["author"].str.contains(r"J\.", case=False, na=False)
            | books["author"].str.contains("John",  case=False, na=False)
        )
    )
    _is_lotr_title = (
        books["title"].str.contains("lord of the rings",    case=False, na=False)
        | books["title"].str.contains("fellowship of the ring", case=False, na=False)
        | books["title"].str.contains("two towers",         case=False, na=False)
        | books["title"].str.contains("return of the king", case=False, na=False)
        | books["title"].str.contains("hobbit",             case=False, na=False)
    )
    lotr_isbns = books[_is_tolkien & _is_lotr_title]["isbn"].tolist()
    print(f"Tolkien ISBNs found: {len(lotr_isbns)}")
    return (lotr_isbns,)


@app.cell
def lotr_eda_stats(books, explicit, lotr_isbns, px):
    # EDA-level view: use the full explicit set (no quality filter yet)
    _lotr_books = (
        books[books["isbn"].isin(lotr_isbns)][["isbn", "title", "author", "year"]]
        .copy()
    )
    _counts = (
        explicit[explicit["isbn"].isin(lotr_isbns)]
        .groupby("isbn").size().rename("n_ratings").reset_index()
    )
    _lotr_books = (
        _lotr_books.merge(_counts, on="isbn", how="left")
        .fillna({"n_ratings": 0})
        .astype({"n_ratings": int})
        .sort_values("n_ratings", ascending=False)
    )

    lotr_explicit     = explicit[explicit["isbn"].isin(lotr_isbns)].copy()
    lotr_raters_all   = lotr_explicit["user_id"].unique()

    print(f"LOTR editions  : {len(_lotr_books)}")
    print(f"Total explicit ratings: {_lotr_books['n_ratings'].sum()}")
    print(f"Unique raters  : {len(lotr_raters_all)}")
    print()
    print(_lotr_books.head(10).to_string(index=False))

    _fig = px.histogram(
        lotr_explicit, x="rating", nbins=10,
        title="LOTR rating distribution (explicit only)",
        labels={"rating": "Rating (1–10)"},
        color_discrete_sequence=["#6a4c93"],
    )
    _fig
    return lotr_explicit, lotr_raters_all


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Preview: what do LOTR readers also read?

    Before any modelling — the simplest possible signal.
    Users who rated LOTR: what other books did they rate most?
    """)
    return


@app.cell
def _(books, explicit, lotr_isbns, lotr_raters_all):
    _other = explicit[
        explicit["user_id"].isin(lotr_raters_all)
        & ~explicit["isbn"].isin(lotr_isbns)
    ].merge(books[["isbn", "title", "author"]], on="isbn", how="left")

    _preview = (
        _other.groupby(["isbn", "title", "author"])
        .agg(n_raters=("user_id", "nunique"), mean_rating=("rating", "mean"))
        .reset_index()
        .sort_values("n_raters", ascending=False)
    )
    print("Top 15 books among LOTR readers (full dataset, no filter):\n")
    print(_preview.head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — MODELS
# ═════════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Part 2 — Models

    Four progressively more sophisticated models, all answering the same question:
    **"I liked Lord of the Rings — what should I read next?"**

    | # | Model | Core idea |
    |---|---|---|
    | 1 | **Popularity baseline** | Books most read by LOTR readers |
    | 2 | **User-based CF** | Find users similar to LOTR readers; use their ratings |
    | 3 | **Item-based CF** | Find books whose rating patterns resemble LOTR |
    | 4 | **SVD** | Decompose the full matrix into latent "taste dimensions" |
    """)
    return


# ── FILTERING ─────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Filtering

    We apply a **single-pass filter**: remove users below the threshold, then books.
    Iterating to convergence collapses to empty on this dataset — the power-law
    distribution means even active users are spread too thin across the long tail.

    This is a **quality vs. coverage** trade-off: tighter thresholds → more reliable
    signal but fewer books visible to the model.
    """)
    return


@app.cell
def filter_data(explicit, MIN_USER_RATINGS, MIN_BOOK_RATINGS):
    _u = explicit["user_id"].value_counts()
    _active = explicit[explicit["user_id"].isin(_u[_u >= MIN_USER_RATINGS].index)]
    _b = _active["isbn"].value_counts()
    filtered = _active[_active["isbn"].isin(_b[_b >= MIN_BOOK_RATINGS].index)].copy()

    print(f"After filtering  (≥{MIN_USER_RATINGS}/user, ≥{MIN_BOOK_RATINGS}/book):")
    print(f"  Ratings : {len(filtered):>8,}  (was {len(explicit):,})")
    print(f"  Users   : {filtered['user_id'].nunique():>8,}")
    print(f"  Books   : {filtered['isbn'].nunique():>8,}")
    return (filtered,)


@app.cell
def lotr_in_filtered(filtered, lotr_isbns):
    lotr_raters = filtered[filtered["isbn"].isin(lotr_isbns)]["user_id"].unique()
    print(f"LOTR raters surviving filter: {len(lotr_raters)}")
    return (lotr_raters,)


# ── MODEL 1: POPULARITY BASELINE ─────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 1 — Popularity Baseline

    Among users who rated a LOTR book, count how many also rated each other book.
    Sort by count; use mean rating as tiebreaker.

    This is the **floor** — any fancier model should beat this.
    If it doesn't, something is wrong with the model.
    """)
    return


@app.cell
def model_baseline(filtered, lotr_isbns, lotr_raters, books, TOP_N):
    baseline = (
        filtered[
            filtered["user_id"].isin(lotr_raters)
            & ~filtered["isbn"].isin(lotr_isbns)
        ]
        .groupby("isbn")
        .agg(n_readers=("user_id", "nunique"), mean_rating=("rating", "mean"))
        .reset_index()
        .query("n_readers >= 5")
        .sort_values("n_readers", ascending=False)
        .merge(books[["isbn", "title", "author"]], on="isbn", how="left")
        .dropna(subset=["title"])
    )
    baseline["mean_rating"] = baseline["mean_rating"].round(2)

    print(f"Popularity Baseline — Top {TOP_N}:\n")
    print(baseline[["title", "author", "n_readers", "mean_rating"]].head(TOP_N).to_string(index=False))
    return (baseline,)


# ── BUILD PIVOT (shared by models 2, 3, 4) ───────────────────────────────────

@app.cell
def build_pivot(filtered):
    pivot = filtered.pivot_table(index="user_id", columns="isbn", values="rating")

    # Mean-centre per user — removes rating-scale bias between users
    _user_means     = pivot.mean(axis=1)
    pivot_centered  = pivot.sub(_user_means, axis=0)

    print(f"Pivot matrix     : {pivot.shape[0]:,} users × {pivot.shape[1]:,} books")
    print(f"Mean user rating : {_user_means.mean():.2f} ± {_user_means.std():.2f}")
    return pivot, pivot_centered


# ── MODEL 2: USER-BASED CF ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 2 — User-Based Collaborative Filtering (Pearson)

    1. Build the "LOTR reader profile" — average centred-rating vector of all LOTR readers
    2. Compute **Pearson correlation** between this profile and every user
       (pairwise: only over books both have rated)
    3. Take top-K neighbours; predict a score for each book:

    $$\hat{r}_{book} = \frac{\sum_{u \in N} \text{sim}(u) \cdot r_{u,book}}{\sum_{u \in N} |\text{sim}(u)|}$$

    **Why Pearson?** It captures relative preferences, not absolute scores —
    a generous rater and a strict rater with the same taste rank high correlation.
    """)
    return


@app.cell
def model_ubcf(pivot_centered, lotr_raters, lotr_isbns, books, TOP_N):
    lotr_profile = (
        pivot_centered
        .loc[pivot_centered.index.isin(lotr_raters)]
        .mean(axis=0)
    )

    similarities = (
        pivot_centered
        .corrwith(lotr_profile, axis=1)
        .dropna()
        .sort_values(ascending=False)
    )

    _top_users   = similarities.head(100)
    _top_ratings = pivot_centered.loc[_top_users.index]
    _weighted    = _top_ratings.multiply(_top_users, axis=0)
    _abs_weights = _top_ratings.notna().astype(float).multiply(_top_users.abs(), axis=0)

    _scores   = _weighted.sum(axis=0, skipna=True) / _abs_weights.sum(axis=0)
    _n_raters = _top_ratings.notna().sum(axis=0)

    _scores = _scores[_n_raters >= 2]
    _scores = _scores.drop(index=[i for i in lotr_isbns if i in _scores.index])
    _scores = _scores.sort_values(ascending=False)

    ubcf = (
        _scores.head(TOP_N).reset_index()
    )
    ubcf.columns = ["isbn", "score"]
    ubcf["score"] = ubcf["score"].round(3)
    ubcf = ubcf.merge(books[["isbn", "title", "author"]], on="isbn", how="left")

    print(f"User-Based CF — Top {TOP_N}:\n")
    print(ubcf[["title", "author", "score"]].to_string(index=False))
    return lotr_profile, similarities, ubcf


# ── MODEL 3: ITEM-BASED CF ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 3 — Item-Based Collaborative Filtering (Cosine)

    Transpose the matrix so rows are **books**, columns are users.
    Compute **cosine similarity** between the LOTR vector and every other book.

    **Why cosine (not Pearson) here?**
    Book vectors are extremely sparse — most users haven't rated most books.
    Treating missing entries as 0 (neutral) and using cosine is more stable
    than Pearson over a mostly-zero vector.

    **Why item-based often beats user-based:**
    Books don't change their rating patterns over time; users do.
    The similarity matrix can also be precomputed once and reused.
    """)
    return


@app.cell
def model_ibcf(pivot_centered, lotr_isbns, books, np, TOP_N):
    import pandas as _pd

    _item_matrix  = pivot_centered.T.fillna(0).values
    _book_index   = pivot_centered.columns.tolist()
    _lotr_indices = [_book_index.index(i) for i in lotr_isbns if i in _book_index]
    _lotr_vec     = _item_matrix[_lotr_indices].mean(axis=0)

    _norms    = np.linalg.norm(_item_matrix, axis=1)
    _lotr_nrm = np.linalg.norm(_lotr_vec)
    _sims     = _item_matrix @ _lotr_vec / (_norms * _lotr_nrm + 1e-10)

    _lotr_set = set(lotr_isbns)
    ibcf = _pd.DataFrame(
        [((_book_index[i], float(_sims[i]))) for i in np.argsort(_sims)[::-1]
         if _book_index[i] not in _lotr_set][:TOP_N],
        columns=["isbn", "cosine_sim"],
    )
    ibcf["cosine_sim"] = ibcf["cosine_sim"].round(4)
    ibcf = ibcf.merge(books[["isbn", "title", "author"]], on="isbn", how="left")

    print(f"Item-Based CF — Top {TOP_N}:\n")
    print(ibcf[["title", "author", "cosine_sim"]].to_string(index=False))
    return (ibcf,)


# ── MODEL 4: SVD ──────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 4 — SVD (Matrix Factorisation)

    Decompose the rating matrix **R** (users × books) into:

    $$R \approx U \cdot \Sigma \cdot V^\top$$

    - **U** (users × k): each user's position in "taste space"
    - **Σ** (k × k): importance of each taste dimension
    - **V** (books × k): each book's position in "taste space"

    We use k = 50 latent dimensions. Nobody names them — they emerge from the data.
    Think of them as abstract axes like "dark epic fantasy" or "short literary fiction".

    **Key advantage over CF:** SVD generalises across the entire dataset.
    A user with only 3 ratings still contributes to the latent factors.
    CF requires direct overlap between users or items.
    """)
    return


@app.cell
def model_svd(pivot_centered, lotr_isbns, books, np, svds, K_LATENT, TOP_N):
    import pandas as _pd
    from scipy.sparse import csr_matrix as _csr

    _matrix = pivot_centered.fillna(0).values.astype(np.float32)
    _U, _sigma, _Vt = svds(_csr(_matrix), k=K_LATENT)

    book_embeddings = (np.diag(_sigma) @ _Vt).T   # (n_books, k)
    book_index      = pivot_centered.columns.tolist()

    _lotr_idx = [book_index.index(i) for i in lotr_isbns if i in book_index]
    lotr_vec  = book_embeddings[_lotr_idx].mean(axis=0)

    print(f"Matrix shape     : {_matrix.shape}")
    print(f"Latent factors k : {K_LATENT}")
    print(f"LOTR editions in filtered matrix: {len(_lotr_idx)}")

    _norms    = np.linalg.norm(book_embeddings, axis=1)
    _lotr_nrm = np.linalg.norm(lotr_vec)
    _sims     = book_embeddings @ lotr_vec / (_norms * _lotr_nrm + 1e-10)

    _lotr_set = set(lotr_isbns)
    svd_recs = _pd.DataFrame(
        [(book_index[i], float(_sims[i])) for i in np.argsort(_sims)[::-1]
         if book_index[i] not in _lotr_set][:TOP_N],
        columns=["isbn", "svd_score"],
    )
    svd_recs["svd_score"] = svd_recs["svd_score"].round(4)
    svd_recs = svd_recs.merge(books[["isbn", "title", "author"]], on="isbn", how="left")

    print(f"\nSVD — Top {TOP_N}:\n")
    print(svd_recs[["title", "author", "svd_score"]].to_string(index=False))
    return book_embeddings, book_index, lotr_vec, svd_recs


# ── OPEN LIBRARY ENRICHMENT ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Open Library Enrichment

    The dataset has no genre information. We fetch subject tags from the
    [Open Library API](https://openlibrary.org) — free, no key required.

    This lets us ask: *do our models recommend books in the same genre as LOTR?*
    Rate-limited to 1 req/s.
    """)
    return


@app.cell
def enrich(svd_recs, requests, time):
    def _fetch_subjects(isbn, max_subjects=5):
        try:
            r    = requests.get(
                f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data",
                timeout=8,
            )
            data = r.json()
            subs = data.get(f"ISBN:{isbn}", {}).get("subjects", [])
            return [s["name"] if isinstance(s, dict) else s for s in subs[:max_subjects]]
        except Exception:
            return []

    _subjects = {}
    for _isbn in svd_recs["isbn"].head(10).tolist():
        _subjects[_isbn] = _fetch_subjects(_isbn)
        time.sleep(1.0)

    svd_enriched = svd_recs.copy()
    svd_enriched["subjects"] = svd_enriched["isbn"].map(
        lambda x: ", ".join(_subjects.get(x, ["—"]))
    )

    print("SVD recommendations + Open Library subjects:\n")
    for _, _r in svd_enriched.head(10).iterrows():
        print(f"  {_r['title'][:42]:<42}  {_r['subjects'][:70]}")
    return (svd_enriched,)


# ── MODEL COMPARISON ──────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Comparing all four models

    Where models **agree** we have higher confidence.
    Where they **disagree** it reveals the strengths and blind spots of each approach.
    """)
    return


@app.cell
def compare(baseline, ubcf, ibcf, svd_recs, go, TOP_N):
    import pandas as _pd

    def _ranks(df, n=TOP_N):
        return {str(t)[:45]: i + 1 for i, t in enumerate(df["title"].dropna().head(n))}

    _b, _u, _i, _s = _ranks(baseline), _ranks(ubcf), _ranks(ibcf), _ranks(svd_recs)
    _all = sorted(set(_b) | set(_u) | set(_i) | set(_s))

    comparison_df = _pd.DataFrame({
        "Book"    : _all,
        "Baseline": [_b.get(t) for t in _all],
        "User CF" : [_u.get(t) for t in _all],
        "Item CF" : [_i.get(t) for t in _all],
        "SVD"     : [_s.get(t) for t in _all],
    })
    comparison_df["n_models"] = comparison_df[["Baseline","User CF","Item CF","SVD"]].notna().sum(axis=1)
    comparison_df = comparison_df.sort_values(["n_models","Baseline"], ascending=[False, True])

    _models = ["Baseline", "User CF", "Item CF", "SVD"]
    _z      = comparison_df[_models].values.tolist()
    _text   = [[str(int(v)) if v is not None and v == v else "—" for v in row] for row in _z]

    fig_compare = go.Figure(go.Heatmap(
        z=_z, x=_models, y=comparison_df["Book"].tolist(),
        text=_text, texttemplate="%{text}",
        colorscale="RdYlGn_r", showscale=True,
        colorbar_title="Rank", zmin=1, zmax=TOP_N,
    ))
    fig_compare.update_layout(
        title  = "Rank of each book across models (green = #1, grey = absent)",
        height = max(400, len(_all) * 22),
        yaxis  = dict(autorange="reversed"),
        margin = dict(l=300),
    )
    fig_compare
    return comparison_df, fig_compare


@app.cell
def _(mo):
    mo.md(r"""
    ## Observations & Limitations

    **What works well:**
    - All models agree on Harry Potter and classic fantasy — a sanity check
    - Item-based CF surfaces genre-specific titles (Pratchett, Gaiman, Gibson, Le Guin)
    - SVD finds books in the same "taste space" even without direct user overlap

    **Known dataset artefacts:**
    - **Wild Animus** — the author mailed free copies to thousands of people who logged
      but rated it poorly. A real system would suppress it using purchase/engagement signals.
    - **Positivity bias** — users mostly rate books they enjoyed; "bad" signal is sparse.
    - **Harry Potter in every model** — it's popular among everyone, not specifically LOTR
      readers. A better baseline would normalise by overall book popularity.

    **Fundamental limitations:**
    - **Popularity bias**: the long tail is invisible — niche books have too few ratings
    - **Cold start**: a new book with 0 ratings cannot appear in any CF recommendation
    - **No temporal signal**: no dates in the dataset — "read right after LOTR" vs
      "read 10 years later" look identical
    - **184 LOTR raters** in the filtered set is a thin signal

    **What I would build with more time:**
    - **Hybrid model**: blend CF scores with content similarity (genre, author, year)
      — solves cold start for new books
    - **Bayesian mean**: shrink small-sample averages toward the global mean
    - **BPR (Bayesian Personalised Ranking)**: optimise for ranking directly,
      not rating prediction
    - **Evaluation**: time-split (train on older, test on newer) + Precision@K / NDCG
    """)
    return


# ── INTERACTIVE DEMO ──────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Demo

    Pick any book from the 300 most-rated titles in the filtered dataset.
    Recommendations are SVD-based — cosine similarity in latent taste space.
    """)
    return


@app.cell
def demo_controls(mo, filtered, books, book_embeddings, book_index):
    _top_isbns = (
        filtered[filtered["isbn"].isin(book_index)]
        .groupby("isbn").size()
        .sort_values(ascending=False)
        .head(300).index.tolist()
    )
    _meta = books[books["isbn"].isin(_top_isbns)][["isbn","title","author"]].drop_duplicates("isbn")
    _options = {
        f"{r['title'][:55]} — {r['author'][:25]}": r["isbn"]
        for _, r in _meta.iterrows()
    }
    demo_picker = mo.ui.dropdown(options=_options, label="Pick a book:")
    demo_picker
    return (demo_picker,)


@app.cell
def demo_results(demo_picker, book_embeddings, book_index, books, np, mo, TOP_N):
    import pandas as _pd

    _isbn = demo_picker.value
    if _isbn is None or _isbn not in book_index:
        mo.stop(True, mo.md("Select a book above to see recommendations."))

    _title  = books[books["isbn"] == _isbn]["title"].values
    _title  = _title[0] if len(_title) > 0 else _isbn
    _q_vec  = book_embeddings[book_index.index(_isbn)]
    _norms  = np.linalg.norm(book_embeddings, axis=1)
    _sims   = book_embeddings @ _q_vec / (_norms * np.linalg.norm(_q_vec) + 1e-10)

    _df = _pd.DataFrame(
        [(book_index[i], round(float(_sims[i]), 4))
         for i in np.argsort(_sims)[::-1] if book_index[i] != _isbn][:TOP_N],
        columns=["isbn", "similarity"],
    )
    _df = _df.merge(books[["isbn","title","author"]], on="isbn", how="left")
    _df.index = range(1, len(_df) + 1)

    mo.vstack([
        mo.md(f"### SVD recommendations for: *{_title}*"),
        mo.ui.table(_df[["title","author","similarity"]]),
    ])
    return


if __name__ == "__main__":
    app.run()
