export interface TaskExecuted {
  tool: string;
  task_id: string;
  success: boolean;
  error?: string;
  output?: string;
}

export interface AgentResponse {
  answer: string;
  latency_ms: number;
  tasks_executed: TaskExecuted[];
  session_id: string;
}

export interface CsvSummary {
  rows: number;
  columns: number;
  numeric_columns: string[];
  categorical_columns: string[];
  missing_values: Record<string, number>;
}

export interface ChartConfig {
  id: string;
  title: string;
  type: 'pie' | 'bar' | 'line' | 'scatter' | 'geo';
  labels?: string[];
  datasets?: any[];
  // for geo maps specifically, there might be other fields, but we'll use 'any' if not known
  map_data?: any; 
}

export interface CsvAnalysisResponse {
  summary: CsvSummary;
  charts: ChartConfig[];
  insights?: string;
  session_id?: string;
}

export interface MemoryInfo {
  count: number;
  cleared?: boolean;
}
