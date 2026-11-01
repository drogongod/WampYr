"""THE FLY'S TRICK, ported: a FIXED, never-trained random sparse projection + winner-take-all
(modeled directly on the real fly olfactory circuit -- ~6 random PN inputs per Kenyon cell, top
~5% survive a global inhibitory WTA -- Dasgupta, Stevens & Navlakha 2017, Science, proved this is a
valid locality-sensitive hash) used to narrow BrickCluster.route()'s candidate set BEFORE the exact
margin computation.

THE GAP THIS FILLS: BrickCluster already has a two-level fast path (_route_fast / _rebuild_regions,
brick_cluster_charlm.py:615-665) -- but it's k-means over the REAL brick population, built only
from consolidate(), and _spawn() marks it stale (_region_fresh=False) the instant any new brick
appears. Until the next consolidate(), and below ROUTE_REGION_INDEX_MIN_BRICKS=400 bricks always,
route() falls back to the exact O(n_bricks) flat scan. The fly's hash never goes stale, because the
address is a fixed function of the QUERY CONTENT ITSELF, computed once at hash-definition time --
never of the current brick population -- so it works from brick #1 onward, no rebuild, no minimum
population, no k-means.

METHODOLOGY: build one real brick population via the REAL, unmodified route()/learn_line-lite path
(skip the expensive learn_batch training -- this test is ONLY about the routing/indexing structure,
not the neural weights, honest scope). Then, on that frozen population, compare the inherited
route() (ground truth, exact flat scan) against route_with_hash() (the new candidate path) on
held-out queries, at MULTIPLE population sizes along the way -- the same 0-mismatch bar
brick_cluster_charlm.py's own comment claims _route_fast was held to ("0/60 mismatches... at
n_bricks=1000").
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import BrickCluster, FEATURE_DIM, VAR_FLOOR_FRAC


class FlyHashIndex:
    """FOUND AND FIXED a real bug in the first version of this (DIAGNOSED, not guessed): the real
    fly's fanin=6-per-KC model assumes a DENSE input space (all 50 projection neurons carry signal
    for any odor). Our tfidf features are the opposite -- SPARSE over a huge FEATURE_DIM=8192
    trigram-hash space, maybe a few dozen nonzero entries per line. With fanin=6, the chance any
    given KC samples even ONE of those few dozen active dimensions is ~2% -- the overwhelming
    majority of KCs score exactly 0 for every query, "top 5%" degenerates into near-random tie-
    breaking among zeros, and the index narrows nothing (confirmed directly: candidate count equaled
    total brick count on every single query in the first run). FIX: keep "fixed, random, untrained,
    winner-take-all" exactly -- just let every KC see every input dimension (a full sign-random-
    projection, the standard fix for sparse high-dimensional input in the same LSH family), so a KC
    actually accumulates real signal from however many of the input's active dimensions there are,
    instead of usually seeing none of them."""
    def __init__(self, feature_dim, n_kc=2000, top_frac=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.n_kc = n_kc
        self.top_k = max(1, int(n_kc * top_frac))
        self.H = rng.choice(np.array([-1, 1], dtype=np.int8), size=(feature_dim, n_kc))
        self.buckets = {}

    def hash_sparse(self, idx_arr, val_arr):
        scores = val_arr @ self.H[idx_arr, :].astype(np.float64)
        active = np.argpartition(scores, -self.top_k)[-self.top_k:]
        return active

    def add_brick(self, brick_idx, active_kc):
        for k in active_kc:
            self.buckets.setdefault(int(k), set()).add(brick_idx)

    def candidates(self, active_kc, min_overlap):
        """SECOND BUG FOUND AND FIXED (diagnosed, not guessed): unioning every brick that shares
        even ONE active bit with the query is far too permissive at these parameters. Expected
        chance overlap between two INDEPENDENT random top_k=100-of-2000 bit-sets is
        K*K/N = 100*100/2000 = 5 bits -- so almost any two unrelated items share >=1 bit purely by
        chance, and a plain union-of-single-bit-buckets admits nearly the whole population every
        time (confirmed directly: candidate count == total brick count on every query in the
        previous run). Real discrimination needs requiring overlap MEANINGFULLY above that chance
        baseline -- count how many of the query's active bits each candidate brick shares, keep
        only bricks clearing min_overlap (well above the ~5-bit chance floor)."""
        counts = {}
        for k in active_kc:
            for b in self.buckets.get(int(k), ()):
                counts[b] = counts.get(b, 0) + 1
        return {b for b, c in counts.items() if c >= min_overlap}


class HashRouteBrickCluster(BrickCluster):
    """Test-only subclass -- does NOT touch the real file. Adds the fly hash as a parallel,
    comparable candidate-narrowing path alongside the real, untouched route()."""
    def __init__(self, *a, n_kc=2000, top_frac=0.05, min_overlap=20, **kw):
        super().__init__(*a, **kw)
        self.fly_index = FlyHashIndex(FEATURE_DIM, n_kc=n_kc, top_frac=top_frac, seed=self.seed)
        self.min_overlap = min_overlap
        self.last_candidate_count = None

    def _hash_of(self, features):
        idx_arr, val_arr = features
        return self.fly_index.hash_sparse(idx_arr, val_arr)

    def register_spawned(self, brick_idx, features):
        self.fly_index.add_brick(brick_idx, self._hash_of(features))

    def route_with_hash(self, features, for_training):
        if not self.bricks:
            return self._spawn() if for_training else None
        gvar = self._global_var()
        neg_ld = self.neg.log_density(features, VAR_FLOOR_FRAC, gvar) if self.neg.n >= 2 else 0.0
        active_kc = self._hash_of(features)
        candidates = self.fly_index.candidates(active_kc, self.min_overlap)
        if not candidates:
            candidates = set(range(len(self.bricks)))  # safety net, never silently wrong
        self.last_candidate_count = len(candidates)
        best_idx, best_margin = -1, -np.inf
        for i in candidates:
            m = self.brick_pos[i].log_density(features, VAR_FLOOR_FRAC, gvar) - neg_ld
            if m > best_margin:
                best_margin, best_idx = m, i
        if for_training and best_margin <= 0.0:
            idx = self._spawn()
            self.register_spawned(idx, features)
            self.last_candidate_count = 1
            return idx
        return best_idx


def route_and_update(cluster, line, for_training=True):
    """route() + the content-model updates learn_line() normally does, WITHOUT the expensive
    learn_batch training -- this test is about the routing/indexing structure only."""
    before = len(cluster.bricks)
    features = cluster.tfidf_sparse(line)
    idx = cluster.route(features, for_training=for_training)
    if for_training:
        if len(cluster.bricks) > before:   # a spawn happened at this call
            cluster.register_spawned(idx, features)
        cluster.brick_pos[idx].update(features)
        cluster.neg.update(features)
        cluster._update_doc_freq(line)
    return idx


# Synthetic topic clusters -- distinct vocab pools so real trigram/tfidf content actually separates,
# same spirit as the (now-missing) routing_scale_toy_test.py this module's docstring referenced.
rng = np.random.default_rng(0)
TOPICS = {
    "france": "the capital of france is paris paris is in france france has a capital city called paris",
    "spain": "the capital of spain is madrid madrid is in spain spain has a capital city called madrid",
    "cooking": "add two cups of flour and one cup of sugar mix well then bake at high heat for an hour",
    "weather": "it is raining heavily today with strong wind and the forecast shows more rain tomorrow",
    "chess": "the knight moves to the center of the board threatening the queen and the bishop pair",
    "space": "the rocket launched into orbit carrying a satellite toward the moon and beyond into space",
    "music": "the guitar and drums play a fast rhythm while the singer holds a long high note",
    "cars": "the engine needs new oil and the brakes should be checked before the long highway drive",
    "garden": "the roses need water and the soil should be turned before planting new seeds this spring",
    "history": "the ancient city was ruled by a king who built walls and temples along the river",
}
topic_names = list(TOPICS.keys())


def gen_line(topic, rng):
    words = TOPICS[topic].split()
    n = rng.integers(5, 10)
    pick = rng.choice(words, size=min(n, len(words)), replace=False)
    rng.shuffle(pick)
    return " ".join(pick)


cluster = HashRouteBrickCluster(V=256, D_brick=40, seed=0, n_kc=2000, top_frac=0.05)

print("=== Growing the real population via the REAL, unmodified route(), checking at each stage ===")
print(f"{'n_bricks':>8} | {'mismatches/50':>14} | {'avg candidates (hash)':>22} | {'avg candidates (flat)':>22}")

TOTAL_TRAIN_LINES = 600
CHECK_AT = {5, 10, 20, 40, 80, 150, 300, 500}
for i in range(TOTAL_TRAIN_LINES):
    topic = topic_names[i % len(topic_names)]
    line = gen_line(topic, rng)
    route_and_update(cluster, line, for_training=True)

    if len(cluster.bricks) in CHECK_AT:
        CHECK_AT.discard(len(cluster.bricks))
        mismatches = 0
        hash_cand_sizes = []
        flat_cand_sizes = []
        for _ in range(50):
            qtopic = rng.choice(topic_names)
            qline = gen_line(qtopic, rng)
            features = cluster.tfidf_sparse(qline)
            true_idx = cluster.route(features, for_training=False)
            hash_idx = cluster.route_with_hash(features, for_training=False)
            hash_cand_sizes.append(cluster.last_candidate_count)
            flat_cand_sizes.append(len(cluster.bricks))
            if true_idx != hash_idx:
                mismatches += 1
        print(f"{len(cluster.bricks):>8} | {mismatches:>8}/50      | "
              f"{np.mean(hash_cand_sizes):>22.1f} | {np.mean(flat_cand_sizes):>22.1f}")

print(f"\nFinal population: {len(cluster.bricks)} bricks "
      f"(ROUTE_REGION_INDEX_MIN_BRICKS=400, so the k-means fast path never engages below that -- "
      f"the exact regime the fly-hash fills).")
