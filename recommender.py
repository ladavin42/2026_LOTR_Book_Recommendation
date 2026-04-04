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
    # LOTR Book Recommender — Models

    Four progressively more sophisticated models, all answering:
    **"I liked Lord of the Rings — what should I read next?"**

    | # | Model | Core idea |
    |---|---|---|
    | 1 | **Popularity baseline** | Books most read by LOTR readers |
    | 2 | **User-based CF** | Find users similar to LOTR readers; use their ratings |
    | 3 | **Item-based CF** | Find books whose rating patterns resemble LOTR |
    | 4 | **SVD** | Decompose the full matrix into latent "taste dimensions" |

    Each model builds on what came before. We compare all four at the end.
    """)
    return


# ── IMPORTS ────────────────────────────────────────────────────────────────────

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


# ── DATA LOADING ───────────────────────────────────────────────────────────────

@app.cell
def load_data(pd):
    DATA = "data/"

    # ── tunable constants ──────────────────────────────────────────────────────
    MIN_USER_RATINGS = 20   # drop users with fewer explicit ratings than this
    MIN_BOOK_RATINGS = 20   # drop books with fewer explicit ratings than this
    K_LATENT         = 50   # SVD latent dimensions
    TOP_N            = 15   # how many recommendations to return per model

    _ratings = pd.read_csv(DATA + "Ratings.csv", encoding="latin-1")
    books_raw = pd.read_csv(DATA + "Books.csv",  encoding="latin-1", on_bad_lines="skip")

    _ratings.columns  = ["user_id", "isbn", "rating"]
    books_raw.columns = ["isbn", "title", "author", "year", "publisher",
                         "img_s", "img_m", "img_l"]

    _ratings["isbn"]  = _ratings["isbn"].str.strip()
    books_raw["isbn"] = books_raw["isbn"].str.strip()

    explicit = _ratings[_ratings["rating"] > 0].copy()

    print(f"Explicit ratings loaded: {len(explicit):,}")
    return K_LATENT, MIN_BOOK_RATINGS, MIN_USER_RATINGS, TOP_N, books_raw, explicit


# ── FILTERING ──────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Data Cleaning & Filtering

    92% of books and 82% of users have fewer than 5 explicit ratings.
    A book rated by 2 people cannot be reliably recommended — we have no idea
    whether those 2 people represent general taste or just chance.

    We apply a **single-pass filter**: remove sparse users first, then sparse books.
    One pass is intentional — iterating to convergence on this dataset drives the
    result to empty, because the power-law distribution means even the most active
    users are spread too thin across the long tail.

    This is a classic **quality vs. coverage** trade-off. The thresholds (20/20)
    are a judgement call — worth discussing at the interview.
    """)
    return


@app.cell
def filter_data(explicit, MIN_USER_RATINGS, MIN_BOOK_RATINGS):
    # Single-pass: filter users, then filter books in the result.
    # Iterating to convergence collapses to empty on this dataset
    # (power-law distribution: active users are too spread across the long tail).
    _u = explicit["user_id"].value_counts()
    _by_active_users = explicit[explicit["user_id"].isin(_u[_u >= MIN_USER_RATINGS].index)]
    _b = _by_active_users["isbn"].value_counts()
    filtered = _by_active_users[_by_active_users["isbn"].isin(_b[_b >= MIN_BOOK_RATINGS].index)].copy()

    print(f"After filtering  (≥{MIN_USER_RATINGS}/user, ≥{MIN_BOOK_RATINGS}/book):")
    print(f"  Ratings : {len(filtered):>8,}  (was {len(explicit):,})")
    print(f"  Users   : {filtered['user_id'].nunique():>8,}")
    print(f"  Books   : {filtered['isbn'].nunique():>8,}")
    return (filtered,)


# ── LOTR IDENTIFICATION ────────────────────────────────────────────────────────

@app.cell
def find_lotr(books_raw, filtered):
    _is_tolkien = books_raw["author"].str.contains("Tolkien", case=False, na=False)
    _is_lotr_title = (
        books_raw["title"].str.contains("lord of the rings",    case=False, na=False)
        | books_raw["title"].str.contains("fellowship of the ring", case=False, na=False)
        | books_raw["title"].str.contains("two towers",         case=False, na=False)
        | books_raw["title"].str.contains("return of the king", case=False, na=False)
        | books_raw["title"].str.contains("hobbit",             case=False, na=False)
    )
    lotr_isbns = books_raw[_is_tolkien & _is_lotr_title]["isbn"].tolist()

    # only care about raters in the filtered (quality) set
    lotr_raters = filtered[filtered["isbn"].isin(lotr_isbns)]["user_id"].unique()

    print(f"Tolkien ISBN count           : {len(lotr_isbns)}")
    print(f"LOTR raters (filtered set)   : {len(lotr_raters)}")
    return lotr_isbns, lotr_raters


# ── MODEL 1: POPULARITY BASELINE ──────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 1 — Popularity Baseline

    **Algorithm:**
    1. Collect all ratings by users who rated any LOTR/Tolkien book
    2. For each other book: count how many of these readers rated it
    3. Sort by reader count; use mean rating as secondary sort

    This is the **floor**. Any fancier model should produce more interesting
    results than this. If it doesn't, the model is broken.

    Note: we require ≥5 LOTR readers to have rated a book to filter noise.
    """)
    return


@app.cell
def model_baseline(filtered, lotr_isbns, lotr_raters, books_raw, TOP_N):
    _lotr_reader_ratings = filtered[
        filtered["user_id"].isin(lotr_raters)
        & ~filtered["isbn"].isin(lotr_isbns)
    ]

    baseline = (
        _lotr_reader_ratings
        .groupby("isbn")
        .agg(n_readers=("user_id", "nunique"), mean_rating=("rating", "mean"))
        .reset_index()
        .query("n_readers >= 5")
        .sort_values("n_readers", ascending=False)
        .merge(books_raw[["isbn", "title", "author"]], on="isbn", how="left")
    )
    baseline = baseline.dropna(subset=["title"])   # drop ISBNs missing from books.csv
    baseline["mean_rating"] = baseline["mean_rating"].round(2)

    print(f"Popularity Baseline — Top {TOP_N}:\n")
    print(baseline[["title", "author", "n_readers", "mean_rating"]].head(TOP_N).to_string(index=False))
    return (baseline,)


# ── BUILD PIVOT (shared by models 2, 3, 4) ────────────────────────────────────

@app.cell
def build_pivot(filtered):
    pivot = filtered.pivot_table(index="user_id", columns="isbn", values="rating")

    # Mean-centre each user's ratings.
    # This removes individual rating-scale bias: a user who gives everything 9
    # and one who gives everything 5 are treated equivalently after centring.
    _user_means = pivot.mean(axis=1)
    pivot_centered = pivot.sub(_user_means, axis=0)

    print(f"Pivot matrix     : {pivot.shape[0]:,} users × {pivot.shape[1]:,} books")
    print(f"Mean user rating : {_user_means.mean():.2f} ± {_user_means.std():.2f}")
    return pivot, pivot_centered


# ── MODEL 2: USER-BASED CF ────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 2 — User-Based Collaborative Filtering (Pearson)

    **Algorithm:**
    1. Build the "LOTR reader profile" — average centred-rating vector of all LOTR readers
    2. Compute **Pearson correlation** between this profile and every user
       (pairwise deletion: only computed on books both have rated)
    3. Take the top-K most similar users as neighbours
    4. For each unread book, predict a score:

    $$\hat{r}_{book} = \frac{\sum_{u \in N} \text{sim}(u) \cdot r_{u,book}}{\sum_{u \in N} |\text{sim}(u)|}$$

    **Why Pearson, not cosine?**
    After mean-centring, Pearson captures *relative* preferences — the shape
    of a user's taste profile, not their absolute scale. Two users with different
    average ratings but the same preferences will show high Pearson correlation.
    """)
    return


@app.cell
def model_ubcf(pivot_centered, lotr_raters, lotr_isbns, books_raw, TOP_N):
    # 1. LOTR reader composite (mean of all LOTR reader vectors)
    lotr_profile = (
        pivot_centered
        .loc[pivot_centered.index.isin(lotr_raters)]
        .mean(axis=0)
    )

    # 2. Pearson correlation — pandas handles NaN pairwise automatically
    # corrwith(axis=1) correlates each row (user) against the profile Series
    similarities = (
        pivot_centered
        .corrwith(lotr_profile, axis=1)
        .dropna()
        .sort_values(ascending=False)
    )

    # 3. Top-K neighbours
    K_NEIGHBORS = 100
    top_users = similarities.head(K_NEIGHBORS)

    # 4. Weighted score for each book
    #    numerator  : sum(sim_u * rating_u_book)  for users who rated it
    #    denominator: sum(|sim_u|)                for users who rated it
    _top_ratings = pivot_centered.loc[top_users.index]
    _weighted    = _top_ratings.multiply(top_users, axis=0)
    _abs_weights = _top_ratings.notna().astype(float).multiply(top_users.abs(), axis=0)

    _scores = _weighted.sum(axis=0, skipna=True) / _abs_weights.sum(axis=0)
    _n_raters = _top_ratings.notna().sum(axis=0)

    # require ≥2 neighbours to have rated a book (matrix is small, be lenient)
    _scores = _scores[_n_raters >= 2]
    _scores = _scores.drop(index=[i for i in lotr_isbns if i in _scores.index])
    _scores = _scores.sort_values(ascending=False)

    ubcf = (
        _scores.head(TOP_N)
        .reset_index()
        .rename(columns={0: "score", "isbn": "isbn"})
    )
    ubcf.columns = ["isbn", "score"]
    ubcf["score"] = ubcf["score"].round(3)
    ubcf = ubcf.merge(books_raw[["isbn", "title", "author"]], on="isbn", how="left")

    print(f"User-Based CF — Top {TOP_N}:\n")
    print(ubcf[["title", "author", "score"]].to_string(index=False))
    return lotr_profile, similarities, ubcf


# ── MODEL 3: ITEM-BASED CF ────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 3 — Item-Based Collaborative Filtering (Cosine)

    **Algorithm:**
    1. Transpose the matrix: now rows are **books**, columns are users
    2. Represent LOTR as the mean of all its edition vectors
    3. Compute **cosine similarity** between LOTR and every other book

    **Why cosine here instead of Pearson?**
    For book vectors (which are very sparse — most users haven't rated most books),
    treating missing entries as 0 (neutral) and using cosine similarity is a
    practical choice. Pearson over a mostly-zero vector is unstable.

    **Why item-based can outperform user-based:**
    - More stable: books don't "change their tastes" over time
    - The dense popular books have hundreds of raters → stable, reliable vectors
    - Scales better: precompute the book×book similarity matrix once
    """)
    return


@app.cell
def model_ibcf(pivot_centered, lotr_isbns, books_raw, np, TOP_N):
    # book × user matrix — fill NaN with 0 (unrated = neutral)
    _item_matrix = pivot_centered.T.fillna(0).values  # shape: (n_books, n_users)
    _book_index  = pivot_centered.columns.tolist()

    # LOTR vector: mean across all LOTR editions present in the filtered matrix
    _lotr_indices = [_book_index.index(i) for i in lotr_isbns if i in _book_index]
    _lotr_vec     = _item_matrix[_lotr_indices].mean(axis=0)  # shape: (n_users,)

    # Cosine similarity: dot(book_vec, lotr_vec) / (||book_vec|| * ||lotr_vec||)
    _norms     = np.linalg.norm(_item_matrix, axis=1)
    _lotr_norm = np.linalg.norm(_lotr_vec)
    _sims      = _item_matrix @ _lotr_vec / (_norms * _lotr_norm + 1e-10)

    _lotr_set = set(lotr_isbns)
    _results  = [
        (_book_index[i], float(_sims[i]))
        for i in np.argsort(_sims)[::-1]
        if _book_index[i] not in _lotr_set
    ][:TOP_N]

    import pandas as _pd
    ibcf = _pd.DataFrame(_results, columns=["isbn", "cosine_sim"])
    ibcf["cosine_sim"] = ibcf["cosine_sim"].round(4)
    ibcf = ibcf.merge(books_raw[["isbn", "title", "author"]], on="isbn", how="left")

    print(f"Item-Based CF — Top {TOP_N}:\n")
    print(ibcf[["title", "author", "cosine_sim"]].to_string(index=False))
    return (ibcf,)


# ── MODEL 4: SVD ──────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Model 4 — SVD (Matrix Factorisation)

    **The big idea:** every user and every book lives in the same low-dimensional
    "taste space". Books close to each other in this space have similar audiences.

    We decompose the rating matrix **R** (users × books) into:

    $$R \approx U \cdot \Sigma \cdot V^\top$$

    - **U** (users × k): each user's position in taste space
    - **Σ** (k × k): importance of each taste dimension
    - **V** (books × k): each book's position in taste space

    We choose k = 50. Nobody names the dimensions — they emerge from the data.
    Think of them as abstract axes like "dark epic fantasy" or "short literary fiction".

    **To recommend books similar to LOTR:**
    1. Look up LOTR's vector in **V** (scaled by **Σ**)
    2. Compute cosine similarity to all other book vectors
    3. Return the closest books

    **Key advantage over CF:** SVD generalises across the entire dataset. A user
    who rated 3 books still contributes to the latent factors. CF requires
    direct overlap between users or items.
    """)
    return


@app.cell
def model_svd(pivot_centered, lotr_isbns, books_raw, np, svds, K_LATENT, TOP_N):
    import pandas as _pd
    from scipy.sparse import csr_matrix as _csr

    # Fill NaN with 0 after mean-centring: "no opinion = neutral"
    _matrix = pivot_centered.fillna(0).values.astype(np.float32)
    _sparse = _csr(_matrix)

    # Truncated SVD: only computes the top-k singular vectors
    # Much faster than full SVD — we don't need the small factors
    _U, _sigma, _Vt = svds(_sparse, k=K_LATENT)

    # Book embeddings: shape (n_books, k), scaled by singular values
    # The scaling ensures dot products approximate the original ratings
    book_embeddings = (np.diag(_sigma) @ _Vt).T   # (n_books, k)
    book_index      = pivot_centered.columns.tolist()

    # LOTR embedding: mean of all LOTR edition vectors in latent space
    _lotr_idx = [book_index.index(i) for i in lotr_isbns if i in book_index]
    lotr_vec  = book_embeddings[_lotr_idx].mean(axis=0)

    print(f"Matrix shape     : {_matrix.shape}")
    print(f"Latent factors k : {K_LATENT}")
    print(f"LOTR editions in filtered matrix: {len(_lotr_idx)}")

    # Cosine similarity in latent space
    _norms    = np.linalg.norm(book_embeddings, axis=1)
    _lotr_nrm = np.linalg.norm(lotr_vec)
    _sims     = book_embeddings @ lotr_vec / (_norms * _lotr_nrm + 1e-10)

    _lotr_set = set(lotr_isbns)
    _results  = [
        (book_index[i], float(_sims[i]))
        for i in np.argsort(_sims)[::-1]
        if book_index[i] not in _lotr_set
    ][:TOP_N]

    svd_recs = _pd.DataFrame(_results, columns=["isbn", "svd_score"])
    svd_recs["svd_score"] = svd_recs["svd_score"].round(4)
    svd_recs = svd_recs.merge(books_raw[["isbn", "title", "author"]], on="isbn", how="left")

    print(f"\nSVD — Top {TOP_N}:\n")
    print(svd_recs[["title", "author", "svd_score"]].to_string(index=False))
    return book_embeddings, book_index, lotr_vec, svd_recs


# ── OPEN LIBRARY ENRICHMENT ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Open Library Enrichment

    The dataset has no genre information. We enrich our recommendations
    with subject tags from the [Open Library API](https://openlibrary.org)
    — free, no key required.

    This lets us ask: *do our models recommend books in the same genre as LOTR?*
    It also gives us a quick sanity check on result quality.

    We rate-limit to 1 request/second to be a good API citizen.
    """)
    return


@app.cell
def enrich(svd_recs, requests, time):
    def _fetch_subjects(isbn, max_subjects=5):
        try:
            url = (
                f"https://openlibrary.org/api/books"
                f"?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
            )
            r    = requests.get(url, timeout=8)
            data = r.json()
            key  = f"ISBN:{isbn}"
            if key not in data:
                return []
            subjects = data[key].get("subjects", [])
            return [s["name"] if isinstance(s, dict) else s for s in subjects[:max_subjects]]
        except Exception:
            return []

    _top_isbns   = svd_recs["isbn"].head(10).tolist()
    _subjects    = {}
    for _isbn in _top_isbns:
        _subjects[_isbn] = _fetch_subjects(_isbn)
        time.sleep(1.0)

    svd_enriched = svd_recs.copy()
    svd_enriched["subjects"] = svd_enriched["isbn"].map(
        lambda x: ", ".join(_subjects.get(x, ["—"]))
    )

    print("SVD recommendations + Open Library subjects:\n")
    for _, _row in svd_enriched.head(10).iterrows():
        print(f"  {_row['title'][:42]:<42}  {_row['subjects'][:70]}")
    return (svd_enriched,)


# ── MODEL COMPARISON ──────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Comparing all four models

    Where models **agree**, we have higher confidence.
    Where they **disagree**, the difference is worth exploring — it often
    reveals the strengths and weaknesses of each approach.
    """)
    return


@app.cell
def compare(baseline, ubcf, ibcf, svd_recs, go, TOP_N):
    import pandas as _pd

    # Build a presence/rank matrix: rows = unique books, cols = models
    def _ranks(df, title_col="title", n=TOP_N):
        titles = df[title_col].dropna().head(n)
        return {str(t)[:45]: i + 1 for i, t in enumerate(titles)}

    _b = _ranks(baseline)
    _u = _ranks(ubcf)
    _i = _ranks(ibcf)
    _s = _ranks(svd_recs)

    _all_books = sorted(set(_b) | set(_u) | set(_i) | set(_s))

    comparison_df = _pd.DataFrame({
        "Book"    : _all_books,
        "Baseline": [_b.get(t, None) for t in _all_books],
        "User CF" : [_u.get(t, None) for t in _all_books],
        "Item CF" : [_i.get(t, None) for t in _all_books],
        "SVD"     : [_s.get(t, None) for t in _all_books],
    })

    # Sort by how many models include each book
    comparison_df["n_models"] = comparison_df[["Baseline","User CF","Item CF","SVD"]].notna().sum(axis=1)
    comparison_df = comparison_df.sort_values(["n_models","Baseline"], ascending=[False, True])

    # Heatmap: rank = colour (lower = better), grey = absent
    _models = ["Baseline", "User CF", "Item CF", "SVD"]
    _z      = comparison_df[_models].values.tolist()
    _text   = [
        [str(int(v)) if v == v and v is not None else "—" for v in row]
        for row in _z
    ]

    fig_compare = go.Figure(go.Heatmap(
        z             = _z,
        x             = _models,
        y             = comparison_df["Book"].tolist(),
        text          = _text,
        texttemplate  = "%{text}",
        colorscale    = "RdYlGn_r",
        showscale     = True,
        colorbar_title= "Rank",
        zmin          = 1,
        zmax          = TOP_N,
    ))
    fig_compare.update_layout(
        title  = "Rank of each book across models (green = ranked 1st, grey = not in top list)",
        height = max(400, len(_all_books) * 22),
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
    - All models agree on Harry Potter and classic fantasy — a good sanity check
    - SVD surfaces more diverse recommendations due to latent generalisation
    - Item-based CF is most stable (books don't "change their rating patterns")

    **Known artefacts in this dataset:**
    - **Wild Animus** — the author mailed free copies to thousands of people who
      logged but rated it poorly. It appears in many lists as a false positive.
      Real production systems use signals like purchase data to suppress this.
    - **Positivity bias** — users tend to rate books they already liked; the dataset
      barely captures "books I hated".

    **Fundamental limitations:**
    - **Popularity bias**: the long tail is invisible — niche books have too few
      ratings to appear in any recommendation
    - **Cold start**: a new book with 0 ratings cannot be recommended by any of these
      models. Content-based (genre, author, plot keywords) would handle this.
    - **583 LOTR raters** is a thin signal; a larger dataset would help all models
    - **No temporal signal**: the dataset has no dates — we can't distinguish "I read
      this right after LOTR" from "I read this 10 years later"

    **What I would build with more time:**
    - **Hybrid model**: combine CF scores with content similarity (genre from Open
      Library, author overlap, publication year)
    - **Bayesian mean**: shrink small-sample mean ratings toward the global mean
      to reduce noise from books with few ratings
    - **BPR (Bayesian Personalised Ranking)**: train on pairs (liked/not-liked) rather
      than explicit scores — more natural objective for ranking
    - **Evaluation**: split by time (train on older ratings, test on newer) and
      measure Precision@K, NDCG — so far we have only qualitative validation
    """)
    return


# ── INTERACTIVE DEMO ──────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Demo

    Select any book in the filtered dataset to get SVD-based recommendations.
    The dropdown shows the 300 most-rated books — a representative sample.
    """)
    return


@app.cell
def demo_controls(mo, filtered, books_raw, book_embeddings, book_index):
    # Top 300 most-rated books in the filtered matrix
    _top_isbns = (
        filtered[filtered["isbn"].isin(book_index)]
        .groupby("isbn").size()
        .sort_values(ascending=False)
        .head(300)
        .index.tolist()
    )
    _meta = (
        books_raw[books_raw["isbn"].isin(_top_isbns)][["isbn", "title", "author"]]
        .drop_duplicates("isbn")
    )
    _options = {
        f"{row['title'][:55]} — {row['author'][:25]}": row["isbn"]
        for _, row in _meta.iterrows()
    }

    demo_picker = mo.ui.dropdown(
        options   = _options,
        label     = "Pick a book:",
    )
    demo_picker
    return (demo_picker,)


@app.cell
def demo_results(demo_picker, book_embeddings, book_index, books_raw, np, mo, TOP_N):
    import pandas as _pd

    _isbn = demo_picker.value
    if _isbn is None or _isbn not in book_index:
        mo.stop(True, mo.md("Select a book above to see recommendations."))

    _title = books_raw[books_raw["isbn"] == _isbn]["title"].values
    _title = _title[0] if len(_title) > 0 else _isbn

    _q_vec   = book_embeddings[book_index.index(_isbn)]
    _norms   = np.linalg.norm(book_embeddings, axis=1)
    _q_norm  = np.linalg.norm(_q_vec)
    _sims    = book_embeddings @ _q_vec / (_norms * _q_norm + 1e-10)

    _results = [
        (book_index[i], round(float(_sims[i]), 4))
        for i in np.argsort(_sims)[::-1]
        if book_index[i] != _isbn
    ][:TOP_N]

    _df = _pd.DataFrame(_results, columns=["isbn", "similarity"])
    _df = _df.merge(books_raw[["isbn", "title", "author"]], on="isbn", how="left")
    _df.index = range(1, len(_df) + 1)

    mo.vstack([
        mo.md(f"### SVD recommendations for: *{_title}*"),
        mo.ui.table(_df[["title", "author", "similarity"]]),
    ])
    return


if __name__ == "__main__":
    app.run()
