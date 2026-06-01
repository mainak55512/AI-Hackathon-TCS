import React, { useMemo, useState } from "react";

const mockHistory = [
  {
    id: "FRD-9912",
    status: "HIGH",
    timestamp: "2026-05-12 09:42:11",
    type: "JSON",
  },
  {
    id: "FRD-8831",
    status: "MEDIUM",
    timestamp: "2026-05-12 08:14:03",
    type: "XML",
  },
  {
    id: "FRD-7720",
    status: "LOW",
    timestamp: "2026-05-11 22:18:44",
    type: "JSON",
  },
];

const samplePayload = `{
  "transactionId": "TX-92181",
  "accountId": "ACC-9920",
  "amount": 18000,
  "currency": "USD",
  "riskScore": 92,
  "origin": "UNKNOWN",
  "flags": ["MULTI_LOGIN", "GEO_MISMATCH"]
}`;

function StatusDot({ tone }) {
  const colors = {
    success: "#22c55e",
    danger: "#f87171",
    primary: "#3b82f6",
    neutral: "#94a3b8",
  };

  return (
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: "999px",
        background: colors[tone] || colors.neutral,
        display: "inline-block",
      }}
    />
  );
}

function TerminalIcon() {
  return (
    <div className="ct-icon-box">
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
      >
        <path d="M5 7L10 12L5 17" />
        <path d="M13 17H19" />
      </svg>
    </div>
  );
}

export default function FraudLogAnalyzer() {
  const [payload, setPayload] = useState(samplePayload);

  const parsed = useMemo(() => {
    try {
      const json = JSON.parse(payload);

      return {
        valid: true,
        type: "JSON",
        risk: json.riskScore || "--",
        transaction: json.transactionId || "--",
        flags: json.flags?.length || 0,
      };
    } catch {
      const isXml =
        payload.trim().startsWith("<") && payload.trim().endsWith(">");

      if (isXml) {
        return {
          valid: true,
          type: "XML",
          risk: "--",
          transaction: "--",
          flags: "--",
        };
      }

      return {
        valid: false,
      };
    }
  }, [payload]);

  return (
    <>
      <style>{`
        :root {
          --bg-1: #0b1220;
          --bg-2: #111827;
          --surface-2: rgba(17, 24, 39, 0.92);
          --border: rgba(148, 163, 184, 0.18);
          --text: #e5e7eb;
          --text-muted: #94a3b8;
          --primary: #3b82f6;
          --success: #22c55e;
          --danger: #f87171;
          --shadow: 0 10px 30px rgba(0,0,0,0.35);
          --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }

        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
          font-family: Inter, system-ui, sans-serif;
          color: var(--text);
        }

        .ct-shell {
          display: grid;
          grid-template-columns: 320px 1fr;
          min-height: 100vh;
          background:
            radial-gradient(circle at top right, rgba(59,130,246,0.08), transparent 28%);
        }

        .ct-panel,
        .ct-main-card {
          position: relative;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: 10px;
          box-shadow: var(--shadow);
        }

        .ct-sidebar {
          padding: 20px;
          border-right: 1px solid rgba(255,255,255,0.06);
        }

        .ct-main {
          padding: 20px;
          overflow: auto;
        }

        .ct-panel {
          height: 100%;
          padding: 18px;
        }

        .ct-main-card {
          padding: 24px;
        }

        .corner-mark {
          position: absolute;
          right: -1px;
          bottom: -1px;
          width: 14px;
          height: 14px;
          border-top: 1px solid var(--border);
          border-left: 1px solid var(--border);
        }

        .ct-title-row {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 22px;
        }

        .ct-icon-box {
          width: 44px;
          height: 44px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.06);
          color: var(--primary);
        }

        .ct-icon-box svg {
          width: 22px;
          height: 22px;
        }

        .ct-heading,
        .ct-label,
        .ct-button,
        .ct-history-id,
        .ct-status-text {
          font-family: var(--mono);
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }

        .ct-heading {
          font-size: 13px;
          font-weight: 700;
        }

        .ct-subtitle {
          font-size: 13px;
          color: var(--text-muted);
          margin-top: 4px;
        }

        .ct-label {
          font-size: 11px;
          font-weight: 700;
          color: var(--text-muted);
          margin-bottom: 10px;
        }

        .ct-history {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .ct-history-item {
          position: relative;
          border: 1px solid var(--border);
          border-radius: 10px;
          background: rgba(255,255,255,0.02);
          padding: 14px;
          transition: 0.15s ease;
        }

        .ct-history-item:hover {
          background: rgba(255,255,255,0.04);
          transform: translateY(-1px);
        }

        .ct-history-item[data-tone="primary"] {
          border-left: 2px solid var(--primary);
        }

        .ct-history-item[data-tone="success"] {
          border-left: 2px solid var(--success);
        }

        .ct-history-item[data-tone="neutral"] {
          border-left: 2px solid #64748b;
        }

        .ct-history-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 12px;
        }

        .ct-history-id {
          font-size: 11px;
          font-weight: 700;
        }

        .ct-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .ct-status-text {
          font-size: 10px;
          color: var(--text-muted);
          font-weight: 700;
        }

        .ct-history-meta {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          font-family: var(--mono);
          font-size: 11px;
          color: var(--text-muted);
        }

        .ct-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
          margin-bottom: 22px;
        }

        .ct-stat {
          position: relative;
          background: rgba(255,255,255,0.02);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 18px;
        }

        .ct-stat[data-tone="primary"] {
          border-left: 2px solid var(--primary);
        }

        .ct-stat[data-tone="success"] {
          border-left: 2px solid var(--success);
        }

        .ct-stat[data-tone="neutral"] {
          border-left: 2px solid #64748b;
        }

        .ct-value {
          font-size: 1.8rem;
          font-weight: 800;
          margin-top: 10px;
          font-family: var(--mono);
        }

        .ct-input-wrap {
          position: relative;
          margin-top: 8px;
        }

        .ct-textarea {
          width: 100%;
          min-height: 340px;
          resize: vertical;
          background: rgba(0,0,0,0.28);
          border: 1px solid var(--border);
          border-radius: 10px;
          color: var(--text);
          padding: 16px;
          outline: none;
          font-size: 13px;
          line-height: 1.7;
          font-family: var(--mono);
          transition: 0.15s ease;
        }

        .ct-textarea:focus {
          border-color: rgba(59,130,246,0.5);
          box-shadow: 0 0 0 1px rgba(59,130,246,0.35);
        }

        .ct-actions {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 18px;
        }

        .ct-button {
          border-radius: 6px;
          height: 40px;
          padding: 0 16px;
          border: 1px solid transparent;
          cursor: pointer;
          font-size: 11px;
          font-weight: 700;
          transition: 0.15s ease;
        }

        .ct-button-primary {
          background: var(--primary);
          color: white;
        }

        .ct-button-primary:hover {
          opacity: 0.92;
        }

        .ct-button-secondary {
          background: transparent;
          border-color: var(--border);
          color: var(--text);
        }

        .ct-button-secondary:hover {
          background: rgba(255,255,255,0.04);
        }

        .ct-analysis {
          margin-top: 24px;
          position: relative;
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 18px;
          background: rgba(255,255,255,0.02);
        }

        .ct-analysis[data-tone="success"] {
          border-left: 2px solid var(--success);
        }

        .ct-analysis[data-tone="neutral"] {
          border-left: 2px solid var(--danger);
        }

        .ct-analysis-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 20px;
          margin-top: 18px;
        }

        .ct-analysis-value {
          margin-top: 8px;
          font-size: 1.1rem;
          font-family: var(--mono);
          font-weight: 700;
        }

        @media (max-width: 980px) {
          .ct-shell {
            grid-template-columns: 1fr;
          }

          .ct-grid,
          .ct-analysis-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      <div className="ct-shell">
        <aside className="ct-sidebar">
          <div className="ct-panel">
            <div className="ct-title-row">
              <TerminalIcon />

              <div>
                <div className="ct-heading">FRAUD LOG HISTORY</div>
                <div className="ct-subtitle">
                  MONITORED INGEST EVENTS
                </div>
              </div>
            </div>

            <div className="ct-history">
              {mockHistory.map((item) => (
                <div
                  key={item.id}
                  className="ct-history-item"
                  data-tone={
                    item.status === "HIGH"
                      ? "primary"
                      : item.status === "MEDIUM"
                      ? "success"
                      : "neutral"
                  }
                >
                  <div className="ct-history-top">
                    <div className="ct-history-id">{item.id}</div>

                    <div className="ct-status">
                      <StatusDot
                        tone={
                          item.status === "HIGH"
                            ? "primary"
                            : item.status === "MEDIUM"
                            ? "success"
                            : "neutral"
                        }
                      />

                      <span className="ct-status-text">
                        {item.status}
                      </span>
                    </div>
                  </div>

                  <div className="ct-history-meta">
                    <span>{item.type}</span>
                    <span>{item.timestamp}</span>
                  </div>

                  <div className="corner-mark" />
                </div>
              ))}
            </div>

            <div className="corner-mark" />
          </div>
        </aside>

        <main className="ct-main">
          <div className="ct-main-card">
            <div className="ct-title-row">
              <TerminalIcon />

              <div>
                <div className="ct-heading">FRAUD LOG ANALYZER</div>
                <div className="ct-subtitle">
                  JSON / XML PAYLOAD INSPECTION
                </div>
              </div>
            </div>

            <div className="ct-grid">
              <div className="ct-stat" data-tone="primary">
                <div className="ct-label">INGEST TYPE</div>
                <div className="ct-value">
                  {parsed.valid ? parsed.type : "--"}
                </div>
                <div className="corner-mark" />
              </div>

              <div className="ct-stat" data-tone="success">
                <div className="ct-label">RISK SCORE</div>
                <div className="ct-value">
                  {parsed.valid ? parsed.risk : "--"}
                </div>
                <div className="corner-mark" />
              </div>

              <div className="ct-stat" data-tone="neutral">
                <div className="ct-label">FLAG COUNT</div>
                <div className="ct-value">
                  {parsed.valid ? parsed.flags : "--"}
                </div>
                <div className="corner-mark" />
              </div>
            </div>

            <div className="ct-label">PAYLOAD INPUT</div>

            <div className="ct-input-wrap">
              <textarea
                className="ct-textarea"
                value={payload}
                onChange={(e) => setPayload(e.target.value)}
                placeholder="PASTE JSON OR XML PAYLOAD..."
              />
            </div>

            <div className="ct-actions">
              <button className="ct-button ct-button-primary">
                ANALYZE LOG
              </button>

              <button className="ct-button ct-button-secondary">
                CLEAR INPUT
              </button>
            </div>

            <div
              className="ct-analysis"
              data-tone={parsed.valid ? "success" : "neutral"}
            >
              <div className="ct-label">ANALYSIS STATUS</div>

              <div className="ct-status">
                <StatusDot tone={parsed.valid ? "success" : "danger"} />

                <span className="ct-status-text">
                  {parsed.valid
                    ? "PAYLOAD STRUCTURE VALIDATED"
                    : "INVALID PAYLOAD FORMAT"}
                </span>
              </div>

              <div className="ct-analysis-grid">
                <div>
                  <div className="ct-label">TRANSACTION ID</div>
                  <div className="ct-analysis-value">
                    {parsed.transaction || "--"}
                  </div>
                </div>

                <div>
                  <div className="ct-label">SCHEMA TYPE</div>
                  <div className="ct-analysis-value">
                    {parsed.type || "--"}
                  </div>
                </div>

                <div>
                  <div className="ct-label">ANALYSIS STATE</div>
                  <div className="ct-analysis-value">
                    {parsed.valid ? "ACTIVE" : "REJECTED"}
                  </div>
                </div>
              </div>

              <div className="corner-mark" />
            </div>

            <div className="corner-mark" />
          </div>
        </main>
      </div>
    </>
  );
}
