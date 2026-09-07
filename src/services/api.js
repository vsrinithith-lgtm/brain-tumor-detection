import { API_CONFIG, endpointUrl } from '../config/api'

async function request(url, options = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), API_CONFIG.timeoutMs)
  try {
    const response = await fetch(url, { ...options, signal: controller.signal })
    const text = await response.text()
    let payload
    try { payload = text ? JSON.parse(text) : {} } catch { payload = { message: text } }
    if (!response.ok) throw new Error(payload?.detail || payload?.message || `Backend returned ${response.status}`)
    return payload
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The MRI segmentation processing timed out. Please try again.')
    if (error instanceof TypeError) throw new Error('Unable to connect to the AI backend. Please make sure FastAPI backend is running on http://localhost:8000.')
    throw error
  } finally { clearTimeout(timer) }
}

export async function analyzeMRI(singleFile, gtFile = null, multimodalFiles = null) {
  const formData = new FormData()

  if (multimodalFiles && multimodalFiles.t1c && multimodalFiles.t1 && multimodalFiles.t2 && multimodalFiles.flair) {
    formData.append('file_t1c', multimodalFiles.t1c)
    formData.append('file_t1', multimodalFiles.t1)
    formData.append('file_t2', multimodalFiles.t2)
    formData.append('file_flair', multimodalFiles.flair)
  } else if (singleFile) {
    formData.append(API_CONFIG.uploadField, singleFile)
  } else {
    throw new Error('No MRI file selected.')
  }

  if (gtFile) {
    formData.append(API_CONFIG.gtUploadField, gtFile)
  }

  const payload = await request(endpointUrl(API_CONFIG.analyzeEndpoint), {
    method: API_CONFIG.analyzeMethod,
    body: formData
  })
  
  if (!payload.success) {
    throw new Error(payload.error || 'MRI segmentation processing failed.')
  }
  
  return payload
}

export async function checkHealth() {
  return request(endpointUrl(API_CONFIG.healthEndpoint), { method: 'GET' })
}
