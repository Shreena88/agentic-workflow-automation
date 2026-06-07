import { useState } from 'react';
import { AgentTab } from './components/AgentTab/AgentTab';
import { CsvTab } from './components/CsvTab/CsvTab';
import { Bot, BarChart2, Zap } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState<'agent' | 'csv'>('agent');

  return (
    <>
      <div style={{ textAlign: 'center' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
          <Zap size={36} color="var(--accent-primary)" />
          Agentic Workflow Automation
        </h1>
        <p className="subtitle">Powered by Grok • LangGraph • FAISS • FastAPI</p>
      </div>

      <div className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'agent' ? 'active' : ''}`}
          onClick={() => setActiveTab('agent')}
        >
          <Bot size={18} />
          Agent
        </button>
        <button 
          className={`tab-btn ${activeTab === 'csv' ? 'active' : ''}`}
          onClick={() => setActiveTab('csv')}
        >
          <BarChart2 size={18} />
          CSV Analysis
        </button>
      </div>

      <div style={{ width: '100%', transition: 'all 0.3s ease' }}>
        {activeTab === 'agent' && <AgentTab />}
        {activeTab === 'csv' && <CsvTab />}
      </div>
    </>
  );
}

export default App;
