import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { DropZone } from '../ui/DropZone';
import { AgentService, MemoryService } from '../../services/api';
import type { AgentResponse } from '../../types';
import { Brain, Trash2, Paperclip, X, Zap } from 'lucide-react';

export const AgentTab: React.FC = () => {
  const [query, setQuery] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<'ready' | 'loading' | 'success' | 'error'>('ready');
  const [errorMsg, setErrorMsg] = useState('');
  const [response, setResponse] = useState<AgentResponse | null>(null);
  const [showTrace, setShowTrace] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const handleFiles = (files: FileList) => {
    const newFiles = Array.from(files);
    setAttachedFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const [uploadedFiles, setUploadedFiles] = useState<Set<string>>(new Set());

  const handleRun = async () => {
    if (!query.trim()) return;
    setStatus('loading');
    setErrorMsg('');
    setShowTrace(false);

    try {
      let currentSessionId = sessionId;

      for (const file of attachedFiles) {
        if (!uploadedFiles.has(file.name)) {
          if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
             const uploadRes = await AgentService.uploadFile(file, currentSessionId);
             if (uploadRes.session_id) currentSessionId = uploadRes.session_id;
          }
          setUploadedFiles(prev => new Set(prev).add(file.name));
        }
      }

      setSessionId(currentSessionId);

      const res = await AgentService.runAgent({
        query,
        max_tasks: 5,
        session_id: currentSessionId
      });

      setResponse(res);
      setSessionId(res.session_id);
      setStatus('success');
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || err.message || 'Agent run failed');
      setStatus('error');
    }
  };

  const getMemoryInfo = async () => {
    if (!sessionId) {
      alert("You haven't started a session yet.");
      return;
    }
    try {
      const data = await MemoryService.getMemoryInfo(sessionId);
      alert(`Session Memory: ${data.count} items are indexed.`);
    } catch (err) {
      alert('Failed to fetch memory info.');
    }
  };

  const clearMemory = async () => {
    if (!sessionId) {
      alert("Nothing to clear yet.");
      return;
    }
    if (!window.confirm("Are you sure you want to wipe the agent's memory?")) return;
    try {
      await MemoryService.clearSession(sessionId);
      setSessionId(null);
      setResponse(null);
      setAttachedFiles([]);
      setUploadedFiles(new Set());
      setStatus('ready');
      alert("All session data removed.");
    } catch (err) {
      alert('Failed to clear memory.');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div className="section-title" style={{ margin: 0 }}>Context Upload</div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button onClick={getMemoryInfo} variant="default" style={{ padding: '6px 12px', fontSize: '0.75rem', background: '#4c1d95', border: '1px solid #7c3aed' }} icon={<Brain size={14} />}>
              Memory State
            </Button>
            <Button onClick={clearMemory} variant="danger-outline" style={{ padding: '6px 12px', fontSize: '0.75rem' }} icon={<Trash2 size={14} />}>
              Clear Session
            </Button>
          </div>
        </div>

        <DropZone 
          onFilesSelected={handleFiles} 
          multiple 
          accept=".pdf,.txt,.md,.json,.csv,.xlsx" 
          text="Drag & drop context files here, or click to browse"
          subtext="PDF • TXT • MD • JSON • CSV"
        />

        {attachedFiles.length > 0 && (
          <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {attachedFiles.map((f, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.05)', padding: '8px 12px', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--success)' }}>
                  <Paperclip size={14} /> {f.name}
                </span>
                <button onClick={() => removeFile(i)} style={{ background: 'none', border: 'none', color: 'var(--error)', cursor: 'pointer' }}>
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid var(--border-subtle)' }}>
          <label>Agent Query</label>
          <textarea 
            value={query} 
            onChange={e => setQuery(e.target.value)}
            placeholder="e.g. Summarize the attached report and look for trends..."
            onKeyDown={e => {
              if (e.ctrlKey && e.key === 'Enter') handleRun();
            }}
          />
          <Button 
            variant="primary" 
            onClick={handleRun} 
            disabled={status === 'loading' || !query.trim()}
            style={{ width: '100%', marginTop: '12px', padding: '14px' }}
            icon={<Zap size={18} />}
          >
            {status === 'loading' ? 'Agent is thinking...' : 'Run Agent'}
          </Button>
        </div>

        <div className="status-bar">
          <div className={`dot ${status === 'error' ? 'error' : status === 'loading' ? 'loading' : ''}`}></div>
          <span>{status === 'loading' ? 'Running agent...' : status === 'error' ? 'Failed' : status === 'success' ? `Done · Session: ${sessionId}` : 'Ready'}</span>
        </div>
      </Card>

      {errorMsg && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-sm)', color: 'var(--error)' }}>
          {errorMsg}
        </div>
      )}

      {response && (
        <Card className="fade-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div className="section-title" style={{ margin: 0 }}>Final Answer</div>
            <Button 
              onClick={() => setShowTrace(!showTrace)} 
              variant="default"
              style={{ fontSize: '0.75rem', padding: '4px 10px', background: 'transparent', border: '1px solid var(--accent-primary)', color: 'var(--accent-primary)' }}
            >
              {showTrace ? 'Hide Process' : 'Show Process Summary'}
            </Button>
          </div>

          <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '20px', fontSize: '0.95rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
            {response.answer}
          </div>

          {showTrace && (
            <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px dashed var(--border-subtle)' }}>
              <div className="section-title">Process Summary & Steps</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {response.tasks_executed?.map((task, i) => (
                  <div key={i} style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                      <span className="badge" style={{ background: 'rgba(124, 58, 237, 0.2)', color: '#c084fc' }}>{task.tool}</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{task.task_id}</span>
                      <span style={{ marginLeft: 'auto' }}>{task.success ? '✅' : '❌'}</span>
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxHeight: '120px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
                      {task.error ? `⚠️ ${task.error}` : task.output}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '12px', textAlign: 'right' }}>
            Latency: {Math.round(response.latency_ms)} ms
          </div>
        </Card>
      )}


    </div>
  );
};
