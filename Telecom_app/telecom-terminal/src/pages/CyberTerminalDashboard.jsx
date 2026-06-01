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

  // Trigger evaluation recalculations upon slider changes
  useEffect(() => {
    fetchRecommendations();
  }, []);

// --- PLACE THIS INSIDE YOUR CyberTerminalDashboard RETURN BLOCK ---
if (loading) {
  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: 'var(--bg-main)', 
      display: 'flex', 
      flexDirection: 'column',
      alignItems: 'center', 
      justifyContent: 'center',
      gap: '20px'
    }}>
      <div style={{
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderLeft: '3px solid var(--accent-blue)',
        borderRadius: '10px',
        padding: '40px',
        position: 'relative',
        minWidth: '300px',
        textAlign: 'center',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)'
      }}>
        {/* Monospace Uppercase Header */}
        <h2 style={{
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          fontFamily: 'var(--mono)',
          fontSize: '14px',
          margin: '0 0 20px 0',
          color: 'var(--accent-blue)'
        }}>
          INITIALIZING TELEMETRY PARSING...
        </h2>
        
        {/* Cyber Loading Dot/Bar Indicator */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginBottom: '15px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)', animation: 'pulse 1s infinite ease-in-out' }} />
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)', animation: 'pulse 1s infinite ease-in-out', animationDelay: '0.2s' }} />
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)', animation: 'pulse 1s infinite ease-in-out', animationDelay: '0.4s' }} />
        </div>

        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: '10px',
          fontWeight: 'bold',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em'
        }}>
          QUERYING GROQ LLM MATRIX // PORT 5000
        </span>

        {/* Signature Decorator Corner Mark */}
        <div className="corner-mark" />
      </div>

      {/* Inline animation utility keyframes injected dynamically */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
    </div>
  );
}


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
      console.error("Matrix connection broken:", err);
    } finally {
      setLoading(false);
    }
  };

  // Typography Object Rules Stylesheet mapping
  const styles = {
    headerText: {
      textTransform: 'uppercase',
      letterSpacing: '0.12em',
      fontFamily: 'var(--mono)',
      fontSize: '16px',
      margin: '0 0 15px 0'
    },
    monoLabel: {
      fontFamily: 'var(--mono)',
      fontSize: '11px',
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
      marginBottom: '15px'
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg-main)', padding: '40px' }}>
      
      {/* App Header Banner */}
      <header style={{ borderBottom: '1px solid var(--border)', paddingBottom: '20px', marginBottom: '30px' }}>
        <h1 style={{ ...styles.headerText, fontSize: '24px', margin: 0, color: 'var(--accent-blue)' }}>
          CONTEXTUAL RECOMMENDATION ENGINE
        </h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '5px 0 0 0' }}>
          Real-time user behavioral parsing avoiding text-retrieval latency models.
        </p>
      </header>

      {/* Primary Split Console Panel Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
        
        {/* Left Column: Simulation Controllers */}
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
                    color: '#fff'
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

{/* --- PLACE THIS AT THE BOTTOM OF YOUR SIMULATOR SURFACECARD --- */}
<div style={{ marginTop: '25px' }}>
  <button 
    onClick={fetchRecommendations}
    style={{
      width: '100%',
      backgroundColor: 'var(--accent-blue)',
      border: 'none',
      color: '#ffffff',
      padding: '12px',
      borderRadius: '6px',
      fontFamily: 'var(--mono)',
      fontWeight: 'bold',
      fontSize: '12px',
      textTransform: 'uppercase',
      letterSpacing: '0.12em',
      cursor: 'pointer',
      transition: 'all 0.15s ease',
      boxShadow: '0 2px 4px rgba(59, 130, 246, 0.2)'
    }}
    onMouseEnter={(e) => e.target.style.filter = 'brightness(1.15)'}
    onMouseLeave={(e) => e.target.style.filter = 'none'}
  >
    GET RECOMMENDATIONS
  </button>
</div>
          
          </SurfaceCard>

          {/* Local Analytics Diagnostic Status Card */}
          <SurfaceCard tone="neutral">
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <IconFrame>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" strokeWidth="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
              </IconFrame>
              <div>
                <span style={styles.monoLabel}>ENGINE DIAGNOSTICS LAYER</span>
                <div style={styles.statusText}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-success)' }} />
                  SYSTEM STATUS: OPERATIONAL // PIPELINE SYNCHRONIZED
                </div>
              </div>
            </div>
          </SurfaceCard>
        </div>

        {/* Right Column: Engine Recommendations Engine Outbound Stream */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          <h2 style={{ ...styles.headerText, marginBottom: '5px' }}>RESOLVED PERSONALIZED TARGET SELECTION</h2>
          
          {engineResponse && engineResponse.recommendations.length > 0 ? (
            engineResponse.recommendations.map((offer, idx) => {
              const isTopPick = idx === 0;
              return (
                <SurfaceCard 
                  key={offer.offer_id} 
                  tone={isTopPick ? 'success' : 'neutral'}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span style={{ 
                        ...styles.monoLabel, 
                        color: isTopPick ? 'var(--accent-success)' : 'var(--text-muted)' 
                      }}>
                        {isTopPick ? '▲ HIGHEST MATCH TARGET VALUE' : 'ALTERNATIVE MATCH'}
                      </span>
                      <h3 style={{ ...styles.headerText, fontSize: '18px', margin: '4px 0' }}>{offer.name}</h3>
                      <p style={{ fontSize: '13px', margin: '0 0 15px 0', color: 'var(--text-muted)' }}>{offer.description}</p>
                    </div>
                    
                    <div style={{ textAlign: 'right' }}>
                      <span style={styles.monoLabel}>SCORE PRIORITY</span>
                      <div style={styles.valueLarge}>{offer.score}</div>
                    </div>
                  </div>

                  {/* Contextualized Push Message UI container */}
                  <div style={{ 
                    backgroundColor: 'rgba(255,255,255,0.02)', 
                    padding: '12px', 
                    borderRadius: '6px', 
                    border: '1px dashed var(--border)',
                    marginBottom: '15px'
                  }}>
                    <span style={{ ...styles.monoLabel, fontSize: '10px', color: 'var(--accent-blue)' }}>DYNAMIC TERMINAL NOTIFICATION TEXT</span>
                    <p style={{ margin: 0, fontFamily: 'var(--mono)', fontSize: '12px', color: '#e5e7eb' }}>
                      "{offer.generated_pitch}"
                    </p>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={styles.statusText}>
                      <span style={{ 
                        width: '8px', height: '8px', borderRadius: '50%', 
                        backgroundColor: offer.type === 'ROAMING' ? 'var(--accent-danger)' : 'var(--accent-blue)' 
                      }} />
                      OFFER ID: {offer.offer_id} // CAT: {offer.type}
                    </div>
                    
                    <button style={{
                      backgroundColor: isTopPick ? 'var(--accent-success)' : 'transparent',
                      border: isTopPick ? 'none' : '1px solid var(--border)',
                      color: isTopPick ? '#000' : 'var(--text-main)',
                      padding: '8px 16px',
                      borderRadius: '6px',
                      fontFamily: 'var(--mono)',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      fontSize: '11px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.08em',
                      transition: 'all 0.15s ease'
                    }}>
                      PROVISION PACK (${offer.price})
                    </button>
                  </div>
                </SurfaceCard>
              );
            })
          ) : (
            <SurfaceCard tone="danger">
              <span style={styles.monoLabel}>EXECUTION HALTED</span>
              <p style={{ margin: 0, fontFamily: 'var(--mono)', fontSize: '13px' }}>
                NO ELIGIBLE OFFERS MATCH THE CRITERIA COMPILATION. CHECK TELEMETRY BOUNDS.
              </p>
            </SurfaceCard>
          )}
        </div>

      </div>
    </div>
  );
}
