const env = import.meta.env

export const API_CONFIG = {
  baseUrl: (env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, ''),
  analyzeEndpoint: env.VITE_ANALYZE_ENDPOINT || '/segment',
  healthEndpoint: env.VITE_HEALTH_ENDPOINT || '/health',
  uploadField: env.VITE_UPLOAD_FIELD || 'file',
  gtUploadField: env.VITE_GT_UPLOAD_FIELD || 'gt_file',
  analyzeMethod: (env.VITE_ANALYZE_METHOD || 'POST').toUpperCase(),
  timeoutMs: Number(env.VITE_API_TIMEOUT || 60000),
}

export function endpointUrl(path) {
  return /^https?:\/\//i.test(path) ? path : `${API_CONFIG.baseUrl}/${path.replace(/^\//, '')}`
}
