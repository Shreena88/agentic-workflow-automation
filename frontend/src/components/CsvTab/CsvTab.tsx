import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { DropZone } from '../ui/DropZone';
import { Button } from '../ui/Button';
import { CsvService, MemoryService } from '../../services/api';
import type { CsvAnalysisResponse } from '../../types';
import { Brain, Trash2, Download } from 'lucide-react';
import { ChartViewer } from '../Charts/ChartViewer';
import ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';

export const CsvTab: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'ready' | 'loading' | 'success' | 'error'>('ready');
  const [errorMsg, setErrorMsg] = useState('');
  const [analysis, setAnalysis] = useState<CsvAnalysisResponse | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const handleFileSelect = (files: FileList) => {
    if (files.length > 0) {
      setFile(files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus('loading');
    setErrorMsg('');
    setAnalysis(null);

    try {
      const data = await CsvService.analyzeCsv(file);
      setAnalysis(data);
      if (data.session_id) {
        setSessionId(data.session_id);
      }
      setStatus('success');
    } catch (err: any) {
      console.error(err);
      let msg = 'Analysis failed';
      if (err.response?.data?.detail) {
        if (typeof err.response.data.detail === 'string') {
          msg = err.response.data.detail;
        } else if (Array.isArray(err.response.data.detail)) {
          msg = err.response.data.detail.map((d: any) => `${d.loc.join('.')}: ${d.msg}`).join(', ');
        } else {
          msg = JSON.stringify(err.response.data.detail);
        }
      } else if (err.message) {
        msg = err.message;
      }
      setErrorMsg(msg);
      setStatus('error');
    }
  };

  const checkMemory = async () => {
    if (!sessionId) {
      alert("No active session yet. Please analyze a dataset first.");
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
    if (!window.confirm("Are you sure you want to wipe the session memory and uploaded files?")) return;
    try {
      await MemoryService.clearSession(sessionId);
      setSessionId(null);
      setAnalysis(null);
      setFile(null);
      setStatus('ready');
      alert("All session data removed.");
    } catch (err) {
      alert('Failed to clear memory.');
    }
  };

  const handleExportExcel = async () => {
    if (!analysis) return;
    try {
      const workbook = new ExcelJS.Workbook();
      const mainSheet = workbook.addWorksheet('Summary & Insights');
      
      mainSheet.mergeCells('A1:E1');
      const title = mainSheet.getCell('A1');
      title.value = 'DATA ANALYSIS REPORT';
      title.font = { size: 16, color: { argb: 'FF7C3AED' }, bold: true };
      title.alignment = { horizontal: 'center' };

      mainSheet.addRow(['Generated on:', new Date().toLocaleString()]);
      mainSheet.addRow([]);

      mainSheet.addRow(['EXECUTIVE SUMMARY']).font = { bold: true, size: 12 };
      mainSheet.addRow(['Total Rows', analysis.summary.rows]);
      mainSheet.addRow(['Total Columns', analysis.summary.columns]);
      mainSheet.addRow(['Numeric Columns', analysis.summary.numeric_columns.length]);
      mainSheet.addRow(['Categorical Columns', analysis.summary.categorical_columns.length]);
      mainSheet.addRow([]);

      if (analysis.insights) {
        mainSheet.addRow(['AI INSIGHTS']).font = { bold: true, size: 12 };
        const insightCell = mainSheet.getCell(`A${mainSheet.lastRow!.number + 1}`);
        insightCell.value = analysis.insights;
        insightCell.alignment = { wrapText: true, vertical: 'top' };
        const rowNum = Number(insightCell.row);
        mainSheet.mergeCells(rowNum, 1, rowNum + 15, 5);
      }

      // Add chart sheets
      if (analysis.charts) {
        for (const cfg of analysis.charts) {
          if (cfg.type === 'geo') continue;
          const canvas = document.getElementById(`chart-${cfg.id}`) as HTMLCanvasElement;
          if (canvas) {
            let sheetName = cfg.title.substring(0, 31).replace(/[\\/?*[\]:]/g, '');
            if (!sheetName) sheetName = 'Chart';
            
            let nameCounter = 1;
            let finalName = sheetName;
            while (workbook.getWorksheet(finalName)) {
              finalName = `${sheetName.substring(0, 28)}_${nameCounter}`;
              nameCounter++;
            }
            
            const chartSheet = workbook.addWorksheet(finalName);
            const imageId = workbook.addImage({ base64: canvas.toDataURL('image/png'), extension: 'png' });
            chartSheet.addRow([`Chart: ${cfg.title}`]).font = { italic: true, size: 14, color: { argb: 'FF7C3AED' } };
            chartSheet.addImage(imageId, {
              tl: { col: 0, row: 2 },
              ext: { width: 600, height: 350 }
            });
          }
        }
      }

      const buffer = await workbook.xlsx.writeBuffer();
      saveAs(new Blob([buffer]), `Analysis_Report_${new Date().getTime()}.xlsx`);
    } catch (err: any) {
      alert("Excel Export Error: " + err.message);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div className="section-title" style={{ margin: 0 }}>Upload CSV or Excel</div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button onClick={checkMemory} variant="default" style={{ padding: '6px 12px', fontSize: '0.75rem', background: '#4c1d95', border: '1px solid #7c3aed' }} icon={<Brain size={14} />}>
              Memory
            </Button>
            <Button onClick={clearMemory} variant="danger-outline" style={{ padding: '6px 12px', fontSize: '0.75rem' }} icon={<Trash2 size={14} />}>
              Clear
            </Button>
          </div>
        </div>

        <DropZone 
          onFilesSelected={handleFileSelect} 
          accept=".csv,.xlsx,.xls"
          text="Drop a CSV or Excel file here, or click to browse"
        />

        {file && (
          <div style={{ marginTop: '12px', fontSize: '0.9rem', color: 'var(--success)', fontWeight: 500 }}>
            Selected: {file.name}
          </div>
        )}

        <Button 
          variant="primary" 
          onClick={handleAnalyze} 
          disabled={!file || status === 'loading'}
          style={{ width: '100%', marginTop: '16px', padding: '14px' }}
        >
          {status === 'loading' ? 'Analyzing Dataset...' : 'Analyze Dataset'}
        </Button>

        <div className="status-bar">
          <div className={`dot ${status === 'error' ? 'error' : status === 'loading' ? 'loading' : ''}`}></div>
          <span>{status === 'loading' ? 'Processing...' : status === 'error' ? 'Failed' : status === 'success' ? 'Analysis Complete' : 'Ready'}</span>
        </div>
      </Card>

      {errorMsg && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-sm)', color: 'var(--error)' }}>
          {errorMsg}
        </div>
      )}

      {analysis && (
        <>
          <Card>
            <div className="section-title">Dataset Overview</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '20px' }}>
              <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-primary)' }}>{analysis.summary.rows.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Rows</div>
              </div>
              <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{analysis.summary.columns}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Columns</div>
              </div>
              <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{analysis.summary.numeric_columns.length}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Numeric</div>
              </div>
              <div className="glass-card" style={{ padding: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{analysis.summary.categorical_columns.length}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Categorical</div>
              </div>
            </div>
            
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>NUMERIC COLUMNS</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {analysis.summary.numeric_columns.map(c => (
                  <span key={c} style={{ background: 'rgba(124, 58, 237, 0.2)', color: '#c084fc', padding: '4px 12px', borderRadius: '999px', fontSize: '0.75rem', border: '1px solid rgba(124, 58, 237, 0.3)' }}>{c}</span>
                ))}
              </div>
            </div>
            
            <div style={{ marginTop: '16px' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>CATEGORICAL COLUMNS</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {analysis.summary.categorical_columns.map(c => (
                  <span key={c} style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#93c5fd', padding: '4px 12px', borderRadius: '999px', fontSize: '0.75rem', border: '1px solid rgba(59, 130, 246, 0.3)' }}>{c}</span>
                ))}
              </div>
            </div>
          </Card>

          {analysis.charts && analysis.charts.length > 0 && (
            <Card>
              <div className="section-title">Charts</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
                {analysis.charts.map(chart => (
                  <ChartViewer key={chart.id} config={chart} />
                ))}
              </div>
            </Card>
          )}

          {analysis.insights && (
            <Card>
              <div className="section-title">AI Insights</div>
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.95rem' }}>
                {analysis.insights}
              </div>
            </Card>
          )}

          <Card>
            <Button variant="success" icon={<Download size={18} />} onClick={handleExportExcel} style={{ width: '100%' }}>
              Export Analysis to Excel
            </Button>
          </Card>
        </>
      )}
    </div>
  );
};
