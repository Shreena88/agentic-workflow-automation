import axios from 'axios';

const API_BASE = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE,
});

export const AgentService = {
  runAgent: async (payload: { query: string; max_tasks?: number; session_id?: string | null }) => {
    const { data } = await api.post('/run-agent', payload);
    return data;
  },
  uploadFile: async (file: File, sessionId?: string | null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    const { data } = await api.post('/upload-file', formData);
    return data;
  },
};

export const CsvService = {
  analyzeCsv: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post('/analyze-csv', formData);
    return data;
  },
};

export const MemoryService = {
  getMemoryInfo: async (sessionId: string) => {
    const { data } = await api.get(`/memory/${sessionId}`);
    return data;
  },
  clearSession: async (sessionId: string) => {
    const { data } = await api.delete(`/memory/${sessionId}`);
    return data;
  },
};
