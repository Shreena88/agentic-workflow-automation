import React, { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';

interface DropZoneProps {
  onFilesSelected: (files: FileList) => void;
  accept?: string;
  multiple?: boolean;
  text?: string;
  subtext?: string;
}

export const DropZone: React.FC<DropZoneProps> = ({ 
  onFilesSelected, 
  accept, 
  multiple = false,
  text = 'Drag & drop files here, or click to browse',
  subtext
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
    // reset input so same file can be selected again
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div
      className={`drop-zone ${isDragOver ? 'dragover' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      style={{
        border: `2px dashed ${isDragOver ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
        borderRadius: 'var(--radius-md)',
        padding: '32px',
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'all 0.2s',
        background: isDragOver ? 'rgba(124, 58, 237, 0.05)' : 'transparent',
        color: isDragOver ? 'var(--accent-primary)' : 'var(--text-secondary)'
      }}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleChange}
        accept={accept}
        multiple={multiple}
        style={{ display: 'none' }}
      />
      <UploadCloud size={36} style={{ marginBottom: '12px', opacity: 0.8 }} />
      <div style={{ fontSize: '0.95rem', fontWeight: 500, marginBottom: '6px' }}>
        {text}
      </div>
      {subtext && <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>{subtext}</div>}
    </div>
  );
};
