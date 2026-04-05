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


@app.cell
def imports():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    import time
    from scipy.sparse.linalg import svds

    return go, np, pd, px, svds


@app.cell
def constants():
    DATA             = "data/"
    MIN_USER_RATINGS = 20    # drop users with fewer explicit ratings
    MIN_BOOK_RATINGS = 20    # drop books with fewer explicit ratings
    K_LATENT         = 50    # SVD latent dimensions
    TOP_N            = 15    # recommendations to return per model
    return DATA, K_LATENT, MIN_BOOK_RATINGS, MIN_USER_RATINGS, TOP_N


@app.cell
def load_data(DATA, pd):
    ratings = pd.read_csv(DATA + "Ratings.csv", encoding="latin-1")
    books   = pd.read_csv(DATA + "Books.csv",   encoding="latin-1", on_bad_lines="skip", low_memory=False)
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


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Ratings: explicit vs implicit
    """)
    return


@app.cell
def _(px, ratings):
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
    return


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
    return


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
    mo.md("""
    ## 3 · Sparsity — the core challenge
    """)
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
            f"The rating matrix is **{sparsity*100:.3f}% empty**.  \n"
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
    return


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
    return


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
    return (lotr_raters_all,)


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
    return


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


@app.cell
def _(mo):
    mo.md(r"""
    ## Filtering

    Remove users below the threshold, then books.

    This is a **quality vs. coverage** trade-off: tighter thresholds → more reliable
    signal but fewer books visible to the model.
    """)
    return


@app.cell
def filter_data(MIN_BOOK_RATINGS, MIN_USER_RATINGS, explicit):
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Model 1 — Popularity Baseline

    Among users who rated a LOTR book, count how many also rated each other book.
    Sort by count; use mean rating as tiebreaker.
    """)
    return


@app.cell
def model_baseline(TOP_N, books, filtered, lotr_isbns, lotr_raters):
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


@app.cell
def build_pivot(filtered):
    pivot = filtered.pivot_table(index="user_id", columns="isbn", values="rating")

    # Mean-centre per user — removes rating-scale bias between users
    _user_means     = pivot.mean(axis=1)
    pivot_centered  = pivot.sub(_user_means, axis=0)

    print(f"Pivot matrix     : {pivot.shape[0]:,} users × {pivot.shape[1]:,} books")
    print(f"Mean user rating : {_user_means.mean():.2f} ± {_user_means.std():.2f}")
    return (pivot_centered,)




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
def model_ibcf(TOP_N, books, lotr_isbns, np, pivot_centered):
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

    # store the item matrix and index so the demo can reuse them
    item_matrix  = _item_matrix
    ibcf_book_index = _book_index
    return ibcf, item_matrix, ibcf_book_index


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
def model_svd(K_LATENT, TOP_N, books, lotr_isbns, np, pivot_centered, svds):
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
    return book_embeddings, book_index, svd_recs


# ── CLAUDE'S PICKS ────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Claude's Picks

    A human-curated list: books from within the dataset that any LOTR reader
    should enjoy, chosen on genre knowledge rather than rating patterns.
    Included in the comparison as a reference point for what "correct" looks like.
    """)
    return


@app.cell
def claude_picks(books):
    # (isbn, reason) — ISBNs verified to be in the dataset
    _picks = [
        ("0553262505", "Le Guin — co-invented epic fantasy; same depth as Tolkien"),
        ("0812511816", "Robert Jordan — Wheel of Time is the closest heir to LOTR"),
        ("037582345X", "Pullman — His Dark Materials; world-building on Tolkien's scale"),
        ("0380789035", "Gaiman — American Gods; mythology, epic journey, literary quality"),
        ("0380789019", "Gaiman — Neverwhere; hidden magical world, dark and beautiful"),
        ("0064471047", "C.S. Lewis — Tolkien and Lewis were friends; same DNA"),
        ("0061020710", "Pratchett — Color of Magic; greatest fantasy comedy series"),
        ("0441003257", "Gaiman & Pratchett — Good Omens; two fantasy legends together"),
        ("0312853238", "Orson Scott Card — Ender's Game; different genre, identical audience"),
        ("0345391802", "Douglas Adams — Hitchhiker's Guide; every fantasy reader has read this"),
    ]

    _isbns  = [p[0] for p in _picks]
    _reason = {p[0]: p[1] for p in _picks}
    _order  = {isbn: i for i, isbn in enumerate(_isbns)}

    claude_recs = (
        books[books["isbn"].isin(_isbns)][["isbn", "title", "author"]]
        .drop_duplicates("isbn")
        .copy()
    )
    claude_recs["reason"]  = claude_recs["isbn"].map(_reason)
    claude_recs["_order"]  = claude_recs["isbn"].map(_order)
    claude_recs = claude_recs.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    print("Claude's picks:\n")
    for _, _r in claude_recs.iterrows():
        print(f"  {_r['title'][:55]:<55}  {_r['reason']}")
    return (claude_recs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparing all models

    Where models **agree** we have higher confidence.
    Where they **disagree** it reveals the strengths and blind spots of each approach.
    The **Claude** column is a human-curated reference — genre knowledge, not maths.
    """)
    return


@app.cell
def compare(TOP_N, baseline, claude_recs, go, ibcf, svd_recs):
    import pandas as _pd

    def _ranks(df, n=TOP_N):
        return {str(t)[:45]: i + 1 for i, t in enumerate(df["title"].dropna().head(n))}

    _b, _i, _s, _c = _ranks(baseline), _ranks(ibcf), _ranks(svd_recs), _ranks(claude_recs, n=len(claude_recs))
    _all = sorted(set(_b) | set(_i) | set(_s) | set(_c))

    comparison_df = _pd.DataFrame({
        "Book"    : _all,
        "Baseline": [_b.get(t) for t in _all],
        "Item CF" : [_i.get(t) for t in _all],
        "SVD"     : [_s.get(t) for t in _all],
        "Claude"  : [_c.get(t) for t in _all],
    })
    comparison_df["n_models"] = comparison_df[["Baseline","Item CF","SVD","Claude"]].notna().sum(axis=1)
    comparison_df = comparison_df.sort_values(["n_models","Baseline"], ascending=[False, True])

    _models = ["Baseline", "Item CF", "SVD", "Claude"]
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
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Observations & Limitations

    **What works well:**


    **Known dataset artefacts:**
    - **Positivity bias** — users mostly rate books they enjoyed; "bad" signal is sparse.

    **Fundamental limitations:**
    - **Popularity bias**: the long tail is invisible — niche books have too few ratings
    - **Cold start**: a new book with 0 ratings cannot appear in any CF recommendation
    - **not that many LOTR raters** in the filtered set

    **What I would build with more time:**
    - **Hybrid model**: blend CF scores with content similarity (genre, author, year), perhabs even add user data that is available in the datasets
    - **BPR (Bayesian Personalised Ranking)**: optimise for ranking directly,
      not rating prediction
    - **Evaluation**: time-split (train on older, test on newer) + Precision@K / NDCG
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Demo

    Pick any book from the 300 most-rated titles and a model.
    All four models are available — compare how their recommendations differ.
    """)
    return


@app.cell
def demo_controls(book_index, books, filtered, mo):
    _top_isbns = (
        filtered[filtered["isbn"].isin(book_index)]
        .groupby("isbn").size()
        .sort_values(ascending=False)
        .head(300).index.tolist()
    )
    _meta = (
        books[books["isbn"].isin(_top_isbns)][["isbn","title","author"]]
        .drop_duplicates("isbn")
        .sort_values("title")
    )
    _options = {
        f"{r['title'][:55]} — {r['author'][:25]}": r["isbn"]
        for _, r in _meta.iterrows()
    }
    demo_picker = mo.ui.dropdown(options=_options, label="Book:")
    model_picker = mo.ui.dropdown(
        options=["SVD", "Item CF (Cosine)", "Popularity Baseline", "Claude's Picks"],
        value="SVD",
        label="Model:",
    )
    mo.hstack([demo_picker, model_picker], gap="2rem")
    return demo_picker, model_picker


@app.cell
def demo_results(
    TOP_N, book_embeddings, book_index, books, claude_recs, demo_picker,
    filtered, ibcf_book_index, item_matrix, model_picker, mo, np,
):
    import pandas as _pd

    _isbn  = demo_picker.value
    _model = model_picker.value

    if _isbn is None:
        mo.stop(True, mo.md("Select a book above to see recommendations."))

    _title = books[books["isbn"] == _isbn]["title"].values
    _title = _title[0] if len(_title) > 0 else _isbn

    # ── SVD ──────────────────────────────────────────────────────────────────
    if _model == "SVD":
        if _isbn not in book_index:
            mo.stop(True, mo.md("Book not in filtered matrix — try another."))
        _q   = book_embeddings[book_index.index(_isbn)]
        _sim = book_embeddings @ _q / (np.linalg.norm(book_embeddings, axis=1) * np.linalg.norm(_q) + 1e-10)
        _df  = _pd.DataFrame(
            [(book_index[i], round(float(_sim[i]), 4))
             for i in np.argsort(_sim)[::-1] if book_index[i] != _isbn][:TOP_N],
            columns=["isbn", "score"],
        )
        _df = _df.merge(books[["isbn", "title", "author"]], on="isbn", how="left")

    # ── ITEM CF ───────────────────────────────────────────────────────────────
    elif _model == "Item CF (Cosine)":
        if _isbn not in ibcf_book_index:
            mo.stop(True, mo.md("Book not in filtered matrix — try another."))
        _q   = item_matrix[ibcf_book_index.index(_isbn)]
        _sim = item_matrix @ _q / (np.linalg.norm(item_matrix, axis=1) * np.linalg.norm(_q) + 1e-10)
        _df  = _pd.DataFrame(
            [(ibcf_book_index[i], round(float(_sim[i]), 4))
             for i in np.argsort(_sim)[::-1] if ibcf_book_index[i] != _isbn][:TOP_N],
            columns=["isbn", "score"],
        )
        _df = _df.merge(books[["isbn", "title", "author"]], on="isbn", how="left")

    # ── POPULARITY BASELINE ───────────────────────────────────────────────────
    elif _model == "Popularity Baseline":
        _readers = filtered[filtered["isbn"] == _isbn]["user_id"].unique()
        if len(_readers) == 0:
            mo.stop(True, mo.md("No readers found — try another book."))
        _df = (
            filtered[filtered["user_id"].isin(_readers) & (filtered["isbn"] != _isbn)]
            .groupby("isbn")
            .agg(score=("user_id", "nunique"))
            .reset_index()
            .sort_values("score", ascending=False)
            .head(TOP_N)
            .merge(books[["isbn", "title", "author"]], on="isbn", how="left")
        )

    # ── CLAUDE'S PICKS ────────────────────────────────────────────────────────
    else:
        _df = claude_recs[["isbn", "title", "author", "reason"]].copy()
        _df["score"] = "—"
        _df.index = range(1, len(_df) + 1)
        mo.vstack([
            mo.md(f"### Claude's genre picks *(not query-specific)*"),
            mo.ui.table(_df[["title", "author", "reason"]]),
        ])
        mo.stop()

    _df.index = range(1, len(_df) + 1)
    mo.vstack([
        mo.md(f"### {_model} — recommendations for: *{_title}*"),
        mo.ui.table(_df[["title", "author", "score"]].round(3)),
    ])
    return


if __name__ == "__main__":
    app.run()
