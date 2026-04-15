"use client";

import { useEffect, useMemo, useState } from "react";

type GameItem = {
  item_id: string;
  title: string;
  brand?: string;
  category?: string;
  price?: string;
  description?: string;
  image_url?: string;
  rating?: number;
  rank?: number;
  predicted_rating?: number;
  explanation?: string;
};

type UserHistory = {
  user_id: string;
  history: GameItem[];
};

type UserRecommendations = {
  user_id: string;
  recommendations: GameItem[];
};

type Metrics = {
  dataset: {
    train_rows: number;
    test_rows: number;
    num_users: number;
    num_items: number;
  };
  baseline: {
    mae: number;
    rmse: number;
  };
  item_item_cf: {
    mae: number;
    rmse: number;
    precision_at_10: number;
    recall_at_10: number;
    f1_at_10: number;
    ndcg_at_10: number;
    evaluated_users: number;
  };
  popularity_top_n?: {
    precision_at_10: number;
    recall_at_10: number;
    f1_at_10: number;
    ndcg_at_10: number;
    evaluated_users: number;
  };
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatMetric(value: number) {
  return value.toFixed(4);
}

function itemKey(item: GameItem) {
  return item.item_id;
}

function imageFallback(title: string) {
  return title
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "VG";
}

function GameImage({ item }: { item: GameItem }) {
  if (item.image_url) {
    return <img className="game-image" src={item.image_url} alt={item.title} loading="lazy" />;
  }
  return <div className="image-fallback" aria-label={item.title}>{imageFallback(item.title)}</div>;
}

export default function Home() {
  const [histories, setHistories] = useState<UserHistory[]>([]);
  const [recommendations, setRecommendations] = useState<UserRecommendations[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [selectedUser, setSelectedUser] = useState("");
  const [query, setQuery] = useState("");
  const [selectedItem, setSelectedItem] = useState<GameItem | null>(null);
  const [shortlist, setShortlist] = useState<GameItem[]>([]);

  useEffect(() => {
    async function loadData() {
      const [historyRes, recommendationRes, metricsRes] = await Promise.all([
        fetch("/data/user_history.json"),
        fetch("/data/recommendations.json"),
        fetch("/data/metrics.json")
      ]);
      const historyData: UserHistory[] = await historyRes.json();
      const recommendationData: UserRecommendations[] = await recommendationRes.json();
      const metricsData: Metrics = await metricsRes.json();
      setHistories(historyData);
      setRecommendations(recommendationData);
      setMetrics(metricsData);
      setSelectedUser(recommendationData[0]?.user_id ?? historyData[0]?.user_id ?? "");
    }
    loadData();
  }, []);

  const recommendationMap = useMemo(() => new Map(recommendations.map((entry) => [entry.user_id, entry])), [recommendations]);
  const historyMap = useMemo(() => new Map(histories.map((entry) => [entry.user_id, entry])), [histories]);
  const users = useMemo(() => recommendations.map((entry) => entry.user_id), [recommendations]);
  const filteredUsers = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return users.slice(0, 80);
    return users.filter((user) => user.toLowerCase().includes(normalized)).slice(0, 80);
  }, [query, users]);

  const selectedRecommendations = recommendationMap.get(selectedUser)?.recommendations ?? [];
  const selectedHistory = historyMap.get(selectedUser)?.history ?? [];
  const shortlistIds = useMemo(() => new Set(shortlist.map((item) => itemKey(item))), [shortlist]);

  function chooseUser(user: string) {
    setSelectedUser(user);
    setSelectedItem(null);
  }

  function toggleShortlist(item: GameItem) {
    setShortlist((current) => {
      if (current.some((entry) => itemKey(entry) === itemKey(item))) {
        return current.filter((entry) => itemKey(entry) !== itemKey(item));
      }
      return [item, ...current].slice(0, 6);
    });
  }

  function copyItemId(item: GameItem) {
    navigator.clipboard?.writeText(item.item_id).catch(() => undefined);
  }

  return (
    <main className="app-shell">
      <section className="top-bar">
        <div>
          <p className="eyebrow">Amazon Video Games Review Data</p>
          <h1>Video Game Recommender</h1>
          <p className="intro">Item-item collaborative filtering turns past ratings into ranked game recommendations.</p>
        </div>
        {metrics && (
          <div className="dataset-strip" aria-label="Dataset summary">
            <span>{formatNumber(metrics.dataset.train_rows)} train ratings</span>
            <span>{formatNumber(metrics.dataset.test_rows)} test ratings</span>
            <span>{formatNumber(metrics.dataset.num_items)} games</span>
          </div>
        )}
      </section>

      <section className="workspace">
        <aside className="sidebar">
          <label htmlFor="user-search">Sample user</label>
          <input
            id="user-search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search user ID"
          />
          <div className="user-list" role="listbox" aria-label="Sample users">
            {filteredUsers.map((user) => (
              <button
                key={user}
                className={user === selectedUser ? "user-button active" : "user-button"}
                onClick={() => chooseUser(user)}
              >
                {user}
              </button>
            ))}
          </div>
        </aside>

        <section className="main-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Active user</p>
              <h2>{selectedUser || "Loading users"}</h2>
            </div>
            <div className="score-pill">Click a game for details</div>
          </div>

          <div className="content-grid">
            <section className="recommendation-area" aria-label="Recommendations">
              {selectedRecommendations.map((item) => (
                <button className="game-card" key={`${selectedUser}-${item.item_id}`} onClick={() => setSelectedItem(item)}>
                  <div className="rank">#{item.rank}</div>
                  <GameImage item={item} />
                  <div className="game-copy">
                    <h3>{item.title}</h3>
                    <p>{item.category || item.brand || "Video game"}</p>
                    <div className="game-meta">
                      <span>Predicted {item.predicted_rating?.toFixed(2)}</span>
                      {item.price && <span>{item.price}</span>}
                      {shortlistIds.has(itemKey(item)) && <span>Shortlisted</span>}
                    </div>
                    <p className="explanation">{item.explanation}</p>
                  </div>
                </button>
              ))}
            </section>

            <aside className="insight-panel">
              <section>
                <h2>Highly Rated History</h2>
                <div className="history-list">
                  {selectedHistory.slice(0, 6).map((item) => (
                    <button className="history-row" key={`${selectedUser}-history-${item.item_id}`} onClick={() => setSelectedItem(item)}>
                      <GameImage item={item} />
                      <div>
                        <strong>{item.title}</strong>
                        <span>{item.rating?.toFixed(1)} stars</span>
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              {shortlist.length > 0 && (
                <section>
                  <h2>Demo Shortlist</h2>
                  <div className="shortlist-list">
                    {shortlist.map((item) => (
                      <button key={`shortlist-${item.item_id}`} onClick={() => setSelectedItem(item)}>
                        <strong>{item.title}</strong>
                        <span>{item.predicted_rating ? `Predicted ${item.predicted_rating.toFixed(2)}` : `${item.rating?.toFixed(1)} stars`}</span>
                      </button>
                    ))}
                  </div>
                </section>
              )}

              {metrics && (
                <section>
                  <h2>Model Metrics</h2>
                  <div className="metric-grid">
                    <div><span>Item-CF MAE</span><strong>{formatMetric(metrics.item_item_cf.mae)}</strong></div>
                    <div><span>Baseline MAE</span><strong>{formatMetric(metrics.baseline.mae)}</strong></div>
                    <div><span>Precision@10</span><strong>{formatMetric(metrics.item_item_cf.precision_at_10)}</strong></div>
                    <div><span>Recall@10</span><strong>{formatMetric(metrics.item_item_cf.recall_at_10)}</strong></div>
                    <div><span>NDCG@10</span><strong>{formatMetric(metrics.item_item_cf.ndcg_at_10)}</strong></div>
                    <div><span>Evaluated users</span><strong>{formatNumber(metrics.item_item_cf.evaluated_users)}</strong></div>
                  </div>
                  {metrics.popularity_top_n && (
                    <p className="comparison">
                      Item-CF Precision@10 is {Math.max(1, metrics.item_item_cf.precision_at_10 / Math.max(metrics.popularity_top_n.precision_at_10, 0.000001)).toFixed(1)}x the popularity baseline.
                    </p>
                  )}
                </section>
              )}
            </aside>
          </div>
        </section>
      </section>

      {selectedItem && (
        <div className="detail-backdrop" role="dialog" aria-modal="true" aria-label="Game details" onClick={() => setSelectedItem(null)}>
          <section className="detail-panel" onClick={(event) => event.stopPropagation()}>
            <button className="close-button" onClick={() => setSelectedItem(null)} aria-label="Close details">Close</button>
            <div className="detail-grid">
              <GameImage item={selectedItem} />
              <div>
                <p className="eyebrow">Item details</p>
                <h2>{selectedItem.title}</h2>
                <p className="detail-category">{selectedItem.category || selectedItem.brand || "Video game"}</p>
                <div className="detail-stats">
                  {selectedItem.rank && <span>Rank #{selectedItem.rank}</span>}
                  {selectedItem.predicted_rating && <span>Predicted {selectedItem.predicted_rating.toFixed(2)}</span>}
                  {selectedItem.rating && <span>User rating {selectedItem.rating.toFixed(1)}</span>}
                  {selectedItem.price && <span>{selectedItem.price}</span>}
                </div>
                <p className="detail-explanation">{selectedItem.explanation || "This item appears in the selected user's observed rating history."}</p>
                <dl className="detail-list">
                  <div><dt>Item ID</dt><dd>{selectedItem.item_id}</dd></div>
                  <div><dt>Brand</dt><dd>{selectedItem.brand || "Unknown"}</dd></div>
                  <div><dt>Source</dt><dd>{selectedItem.rank ? "Top-10 recommendation" : "Highly rated history"}</dd></div>
                </dl>
                <div className="action-row">
                  <button className="primary-action" onClick={() => toggleShortlist(selectedItem)}>
                    {shortlistIds.has(itemKey(selectedItem)) ? "Remove from shortlist" : "Add to shortlist"}
                  </button>
                  <button className="secondary-action" onClick={() => copyItemId(selectedItem)}>Copy item ID</button>
                </div>
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
