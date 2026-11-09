import { describe, it, expect, vi, beforeEach } from 'vitest'
import { api, ApiError, NetworkError, setAccessToken } from '@/lib/api'

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setAccessToken(null)
  })

  describe('GET requests', () => {
    it('should make a successful GET request', async () => {
      const mockData = { id: 1, name: 'Test' }
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockData,
      } as Response)

      const result = await api.get('/test')
      expect(result).toEqual(mockData)
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/test',
        expect.objectContaining({
          method: 'GET',
          credentials: 'include',
        })
      )
    })

    it('should include authorization header when token is set', async () => {
      setAccessToken('test-token')
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response)

      await api.get('/test')
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/test',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      )
    })

    it('should append query params', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response)

      await api.get('/test', { params: { page: '1', limit: '10' } })
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/test?page=1&limit=10',
        expect.any(Object)
      )
    })
  })

  describe('POST requests', () => {
    it('should send JSON body', async () => {
      const postData = { name: 'Test Item' }
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 201,
        json: async () => ({ id: 1, ...postData }),
      } as Response)

      const result = await api.post('/items', postData)
      expect(result).toEqual({ id: 1, name: 'Test Item' })
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/items',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(postData),
        })
      )
    })
  })

  describe('Error handling', () => {
    it('should throw ApiError for 4xx/5xx responses', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Not found' }),
      } as Response)

      try {
        await api.get('/not-found')
        // Should not reach here
        expect(true).toBe(false)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError)
        expect((error as ApiError).status).toBe(404)
        expect((error as ApiError).message).toBe('Not found')
      }
    })

    it('should throw NetworkError when offline', async () => {
      Object.defineProperty(navigator, 'onLine', { value: false, writable: true })
      vi.mocked(global.fetch).mockRejectedValueOnce(new Error('Network error'))

      await expect(api.get('/test')).rejects.toThrow(NetworkError)
      Object.defineProperty(navigator, 'onLine', { value: true, writable: true })
    })

    it('should throw NetworkError for connection failures', async () => {
      Object.defineProperty(navigator, 'onLine', { value: true, writable: true })
      vi.mocked(global.fetch).mockRejectedValueOnce(new TypeError('Failed to fetch'))

      await expect(api.get('/test')).rejects.toThrow(NetworkError)
    })
  })

  describe('PATCH requests', () => {
    it('should send partial update', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ id: 1, name: 'Updated' }),
      } as Response)

      await api.patch('/items/1', { name: 'Updated' })
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/items/1',
        expect.objectContaining({
          method: 'PATCH',
        })
      )
    })
  })

  describe('DELETE requests', () => {
    it('should handle 204 No Content', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 204,
        json: async () => ({}),
      } as Response)

      const result = await api.delete('/items/1')
      expect(result).toBeUndefined()
    })
  })
})
