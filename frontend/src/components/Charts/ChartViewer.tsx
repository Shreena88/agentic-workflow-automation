import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { Pie, Bar, Line, Scatter } from 'react-chartjs-2';
import type { ChartConfig } from '../../types';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ChartViewerProps {
  config: ChartConfig;
}

export const ChartViewer: React.FC<ChartViewerProps> = ({ config }) => {
  const mapRef = useRef<HTMLDivElement>(null);

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 } }
      }
    },
    scales: config.type !== 'pie' && config.type !== 'geo' ? {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
    } : undefined
  };

  const chartData = {
    labels: config.labels || [],
    datasets: config.datasets?.map(ds => ({
      ...ds,
      backgroundColor: ds.backgroundColor || [
        'rgba(124, 58, 237, 0.7)',
        'rgba(16, 185, 129, 0.7)',
        'rgba(59, 130, 246, 0.7)',
        'rgba(245, 158, 11, 0.7)',
        'rgba(239, 68, 68, 0.7)'
      ],
      borderColor: ds.borderColor || '#1a1d27',
      borderWidth: 1
    })) || []
  };

  // For Geo Map
  useEffect(() => {
    if (config.type === 'geo' && mapRef.current) {
      const map = L.map(mapRef.current).setView([0, 0], 2);
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CartoDB',
        subdomains: 'abcd',
        maxZoom: 19
      }).addTo(map);

      // Add markers if config has them
      if (config.map_data?.markers) {
        config.map_data.markers.forEach((m: any) => {
          L.circleMarker([m.lat, m.lon], {
            radius: m.val ? Math.max(4, m.val / 10) : 5,
            fillColor: '#7c3aed',
            color: '#c084fc',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.6
          }).bindPopup(m.label || 'Point').addTo(map);
        });
      }

      return () => {
        map.remove();
      };
    }
  }, [config]);

  return (
    <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '16px', fontWeight: 500 }}>
        {config.title}
      </div>
      <div style={{ height: '260px', position: 'relative' }}>
        {config.type === 'pie' && <Pie id={`chart-${config.id}`} data={chartData} options={commonOptions as any} />}
        {config.type === 'bar' && <Bar id={`chart-${config.id}`} data={chartData} options={commonOptions as any} />}
        {config.type === 'line' && <Line id={`chart-${config.id}`} data={chartData} options={commonOptions as any} />}
        {config.type === 'scatter' && <Scatter id={`chart-${config.id}`} data={chartData} options={commonOptions as any} />}
        {config.type === 'geo' && <div id={`chart-${config.id}`} ref={mapRef} style={{ width: '100%', height: '100%', borderRadius: '8px' }} />}
      </div>
    </div>
  );
};
