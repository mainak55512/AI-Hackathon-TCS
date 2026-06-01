import React, { useState, useEffect } from 'react';

// Common SVG Micro-Icon Component Wrapper
const IconFrame = ({ children }) => (
  <div style={{
    width: '44px',
    height: '44px',
    borderRadius: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '1px solid var(--border)'
  }}>
    {children}
  </div>
);

// Unified Surface Card incorporating tone variations and the Corner Notch requirement
const SurfaceCard = ({ children, tone = 'neutral', style = {} }) => {
  let borderLeftColor = 'var(--border)';
  if (tone === 'primary') borderLeftColor = 'var(--accent-blue)';
  if (tone === 'success') borderLeftColor = 'var(--accent-success)';
  if (tone === 'danger') borderLeftColor = 'var(--accent-danger)';

  return (
    <div 
      data-tone={tone}
      style={{
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderLeft: `3px solid ${borderLeftColor}`,
        borderRadius: '10px',
        padding: '20px',
        position: 'relative',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)',
        ...style
      }}
    >
      {children}
      <div className="corner-mark" />
    </div>
  );
};

export default function CyberTerminalDashboard() {
  // Simulator Telemetry States
  const [userId, setUserId] = useState('SUB-2049');
  const [dataUsage, setDataUsage] = useState(65);
  const [dataDepletion, setDataDepletion] = useState(85);
  const [isRoaming, setIsRoaming] = useState(false);
  const [monthlySpend, setMonthlySpend] = useState(50);
  const [historicalPref, setHistoricalPref] = useState('DATA');
  
  // Pipeline Engine Output State
  const [engineResponse, setEngineResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch initial baseline recommendation once upon page load bootstrap
  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          data_usage_gb: dataUsage,
          data_depletion_pct: dataDepletion,
          is_roaming: isRoaming,
          monthly_spend: monthlySpend,
          historical_preference: historicalPref
        })
      });
      const payload = await res.json();
      setEngineResponse(payload);
    } catch (err) {
      console.error("Multi-Agent matrix pipeline connection broken:", err);
    } finally {
      setLoading(false);
    }
  };

  // Typography Rules Stylesheet Map
  const styles = {
    headerText: {
      textTransform: 'uppercase',
      letterSpacing: '0.12em',
      fontFamily: 'var(--mono)',
      fontSize: '14px',
      margin: '0 0 15px 0'
    },
    monoLabel: {
      fontFamily: 'var(--mono)',
      fontSize: '10px',
      fontWeight: 'bold',
      textTransform: 'uppercase',
      color: 'var(--text-muted)',
      letterSpacing: '0.08em',
      display: 'block',
      marginBottom: '6px'
    },
    valueLarge: {
      fontSize: '1.8rem',
      fontWeight: 'bold',
      fontFamily: 'var(--mono)',
      color: 'var(--text-main)'
    },
    statusText: {
      fontFamily: 'var(--mono)',
      fontSize: '11px',
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      textTransform: 'uppercase'
    },
    interactiveInput: {
      width: '100%',
      backgroundColor: 'var(--surface-2)',
      border: '1px solid var(--border)',
      color: 'var(--text-main)',
      padding: '8px',
      borderRadius: '6px',
      fontFamily: 'var(--mono)',
      marginTop: '4px',
      marginBottom: '15px',
      boxSizing: 'border-box'
    }
  };

  // --- CYBER-TERMINAL LOADING SCREEN SNIPPET ---
  if (loading) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg-main)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px' }}>
        <div style={{ backgroundColor: 'var(--bg-surface)', border: '1px solid var(--border)', borderLeft: '3px solid var(--accent-blue)', borderRadius: '10px', padding: '40px', position: 'relative', minWidth: '340px', textAlign: 'center' }}>
          <h2 style={{ textTransform: 'uppercase', letterSpacing: '0.12em', fontFamily: 'var(--mono)', fontSize: '14px', margin: '0 0 20px 0', color: 'var(--accent-blue)' }}>
            EXECUTING MULTI-AGENT SWARM...
          </h2>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginBottom: '15px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)', animation: 'pulse 1s infinite ease-in-out' }} />
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)', animation: 'pulse 1s infinite ease-in-out', animationDelay: '0.2s' }} />
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)', animation: 'pulse 1s infinite ease-in-out', animationDelay: '0.4s' }} />
          </div>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
            OPTIMIZING MARGINS & CREATIVE COPY
          </span>
          <div className="corner-mark" />
        </div>
        <style>{`@keyframes pulse { 0%, 100% { opacity: 0.3; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1.2); }  }`}</style>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg-main)', padding: '40px' }}>
      
      {/* App Header Banner */}
      <header style={{ borderBottom: '1px solid var(--border)', paddingBottom: '20px', marginBottom: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ ...styles.headerText, fontSize: '22px', margin: 0, color: 'var(--accent-blue)' }}>
            CONTEXTUAL MULTI-AGENT RECOMMENDATION COCKPIT
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '5px 0 0 0' }}>
            Distributed microservice orchestration layer integrated with billing, risk matrices, and copy LLMs.
          </p>
        </div>
        {engineResponse && (
          <div style={{ textAlign: 'right', fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--accent-success)' }}>
            PIPELINE LATENCY: {engineResponse.latency_ms} MS
          </div>
        )}
      </header>

      {/* --- TOP ROW: STRATEGIC KPI TRACKING OVERVIEW --- */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '20px', marginBottom: '30px' }}>
        <SurfaceCard tone="success">
          <span style={styles.monoLabel}>CONVERSION RATE ACCELERATION</span>
          <div style={styles.valueLarge}>+24.8%</div>
          <span style={{ ...styles.monoLabel, color: 'var(--accent-success)', marginBottom: 0, marginTop: '4px' }}>AGENT MODEL VS STATIC RULES</span>
        </SurfaceCard>
        <SurfaceCard tone="primary">
          <span style={styles.monoLabel}>ARPU DELTA GROWTH</span>
          <div style={styles.valueLarge}>+$4.12</div>
          <span style={{ ...styles.monoLabel, color: 'var(--accent-blue)', marginBottom: 0, marginTop: '4px' }}>AVERAGE REVENUE PER OPERATOR LINE</span>
        </SurfaceCard>
        <SurfaceCard tone="neutral">
          <span style={styles.monoLabel}>TOTAL SWARM ROUTER CALLS</span>
          <div style={styles.valueLarge}>14.2M</div>
          <span style={{ ...styles.monoLabel, marginBottom: 0, marginTop: '4px' }}>KAFKA STREAM INGESTION RATE</span>
        </SurfaceCard>
        <SurfaceCard tone="neutral">
          <span style={styles.monoLabel}>REVENUE CHURN REDUCTION</span>
          <div style={styles.valueLarge}>-18.3%</div>
          <span style={{ ...styles.monoLabel, marginBottom: 0, marginTop: '4px' }}>PREDICTIVE SYSTEM INTERCEPTIONS</span>
        </SurfaceCard>
      </div>

      {/* Primary Split Console Panel Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
        
        {/* Left Column: Simulation Control Base */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <SurfaceCard tone="primary">
            <h2 style={styles.headerText}>TELEMETRY SIMULATOR INJECTOR</h2>
            
            <label style={styles.monoLabel}>SUBSCRIBER REGISTRY TARGET</label>
            <input 
              style={styles.interactiveInput} 
              type="text" 
              value={userId} 
              onChange={(e) => setUserId(e.target.value)} 
            />

            <label style={styles.monoLabel}>DATA DELETION RATIO ({dataDepletion}%)</label>
            <input 
              type="range" min="0" max="100" style={{ width: '100%', marginBottom: '20px' }}
              value={dataDepletion} onChange={(e) => setDataDepletion(Number(e.target.value))} 
            />

            <label style={styles.monoLabel}>MONTHLY SPEND BENCHMARK (${monthlySpend})</label>
            <input 
              type="range" min="5" max="120" style={{ width: '100%', marginBottom: '20px' }}
              value={monthlySpend} onChange={(e) => setMonthlySpend(Number(e.target.value))} 
            />

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginTop: '10px' }}>
              <div>
                <label style={styles.monoLabel}>ROAMING NETWORK STATUS</label>
                <button 
                  onClick={() => setIsRoaming(!isRoaming)}
                  style={{
                    width: '100%', padding: '10px', borderRadius: '6px', cursor: 'pointer',
                    fontFamily: 'var(--mono)', textTransform: 'uppercase', transition: 'all 0.15s ease',
                    backgroundColor: isRoaming ? 'var(--accent-danger)' : 'transparent',
                    border: isRoaming ? 'none' : '1px solid var(--border)',
                    color: '#fff', fontSize: '11px'
                  }}
                >
                  {isRoaming ? 'ROAMING ACTIVE' : 'DOMESTIC PIPE'}
                </button>
              </div>

              <div>
                <label style={styles.monoLabel}>HISTORICAL PREFERENCE TIER</label>
                <select 
                  style={styles.interactiveInput} 
                  value={historicalPref} 
                  onChange={(e) => setHistoricalPref(e.target.value)}
                >
                  <option value="DATA">DATA DRIVEN</option>
                  <option value="VOICE">VOICE CENTRIC</option>
                </select>
              </div>
            </div>

            {/* --- MANUAL TRIGGER INJECTION ACTION BUTTON --- */}
            <div style={{ marginTop: '25px' }}>
              <button 
                onClick={fetchRecommendations}
                style={{
                  width: '100%', backgroundColor: 'var(--accent-blue)', border: 'none', color: '#ffffff',
                  padding: '12px', borderRadius: '6px', fontFamily: 'var(--mono)', fontWeight: 'bold',
                  fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.12em', cursor: 'pointer',
                  transition: 'all 0.15s ease', boxShadow: '0 2px 4px rgba(59, 130, 246, 0.2)'
                }}
                onMouseEnter={(e) => e.target.style.filter = 'brightness(1.15)'}
                onMouseLeave={(e) => e.target.style.filter = 'none'}
              >
                EXECUTE ENGINE SWARM // GET PERSONALIZATION
              </button>
            </div>
          </SurfaceCard>

          <SurfaceCard tone="neutral">
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <IconFrame>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" strokeWidth="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
              </IconFrame>
              <div>
                <span style={styles.monoLabel}>MULTI-AGENT BROKERING MESH</span>
                <div style={styles.statusText}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-success)' }} />
                  CRM GATEWAY // BILLING API REPLICAS CONNECTED
                </div>
              </div>
            </div>
          </SurfaceCard>
        </div>

        {/* Right Column: Engine Recommendations Engine Outbound Stream */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h2 style={{ ...styles.headerText, marginBottom: '5px' }}>RESOLVED PERSONALIZED TARGET SELECTION</h2>
          
          {engineResponse && engineResponse.recommendations && engineResponse.recommendations.length > 0 ? (
            engineResponse.recommendations.map((offer) => (
              <SurfaceCard key={offer.offer_id} tone="success">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span style={{ ...styles.monoLabel, color: 'var(--accent-success)' }}>
                      ▲ AGENT RANK OPTIMIZATION WINNER
                    </span>
                    <h3 style={{ ...styles.headerText, fontSize: '18px', margin: '4px 0' }}>{offer.name}</h3>
                  </div>
                  
                  <div style={{ textAlign: 'right' }}>
                    <span style={styles.monoLabel}>ACCEPTANCE PROBABILITY</span>
                    <div style={styles.valueLarge}>{offer.p_accept_metric}</div>
                  </div>
                </div>

                {/* Contextualized Push Message UI container */}
                <div style={{ 
                  backgroundColor: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', 
                  border: '1px dashed var(--border)', marginBottom: '15px', marginTop: '10px'
                }}>
                  <span style={{ ...styles.monoLabel, fontSize: '10px', color: 'var(--accent-blue)' }}>
                    LLM GENERATED CREATIVE COPY PITCH
                  </span>
                  <p style={{ margin: 0, fontFamily: 'var(--mono)', fontSize: '12px', color: '#e5e7eb', lineHeight: '1.4' }}>
                    "{offer.generated_pitch}"
                  </p>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={styles.statusText}>
                    <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)' }} />
                    OFFER ID: {offer.offer_id}
                  </div>
                  
                  <button style={{
                    backgroundColor: 'var(--accent-success)', border: 'none', color: '#000',
                    padding: '8px 16px', borderRadius: '6px', fontFamily: 'var(--mono)',
                    fontWeight: 'bold', cursor: 'pointer', fontSize: '11px', textTransform: 'uppercase',
                    letterSpacing: '0.08em', transition: 'all 0.15s ease'
                  }}>
                    PROVISION VIA BILLING RAIL (${offer.price})
                  </button>
                </div>
              </SurfaceCard>
            ))
          ) : (
            <SurfaceCard tone="danger">
              <span style={styles.monoLabel}>EXECUTION HALTED</span>
              <p style={{ margin: 0, fontFamily: 'var(--mono)', fontSize: '13px' }}>
                NO ELIGIBLE OFFERS MATCH THE CRITERIA COMPILATION. CLICK THE GENERATE BUTTON ABOVE TO TRIGGER ENGINE TRACE.
              </p>
            </SurfaceCard>
          )}
        </div>

      </div>
    </div>
  );
}
