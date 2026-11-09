import { AddItemDialog } from '@/components/add-item-dialog'
import { beforeEach, afterEach } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { NextIntlClientProvider } from 'next-intl'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { ItemDetailDialog } from '@/components/item-detail-dialog'
import type { Item } from '@/lib/types'
import { api } from '@/lib/api'
import { toast } from 'sonner'
import { ApplicationAccessContext } from '@/lib/application-access-context'

vi.unmock('next-intl')
vi.mock('@/lib/api', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/api')>(),
  api: { get: vi.fn().mockResolvedValue([]), post: vi.fn(), patch: vi.fn(), delete: vi.fn() },
  setAccessToken: vi.fn(),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }))

function messages(locale: string, namespace: string) {
  return JSON.parse(
    readFileSync(resolve(__dirname, '..', 'messages', locale, `${namespace}.json`), 'utf8'),
  )
}

function wrapperFor(locale: string) {
  const all = Object.fromEntries(
    ['wardrobe', 'constants', 'common'].map((ns) => [ns, messages(locale, ns)]),
  )
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <NextIntlClientProvider locale={locale} messages={all} onError={() => {}}>
          <ApplicationAccessContext.Provider value={{ businessWrite: true, reason: null }}>{children}</ApplicationAccessContext.Provider>
        </NextIntlClientProvider>
      </QueryClientProvider>
    )
  }
}

// The tagger only accepts values from these sets, so each needs a label in constants.
function backendTagValues(name: string): string[] {
  const source = readFileSync(
    resolve(__dirname, '..', '..', 'backend', 'app', 'services', 'ai_service.py'),
    'utf8',
  )
  const start = source.indexOf(`${name} = {`)
  const block = source.slice(start, source.indexOf('}', start))
  return Array.from(block.matchAll(/"([^"]+)"/g), (m) => m[1])
}

const item = {
  id: 'i1',
  type: 'shirt',
  status: 'ready',
  tags: {
    colors: ['navy'],
    pattern: 'striped',
    style: ['preppy'],
    season: ['all-season'],
    fit: 'slim',
  },
  images: [],
  image_url: 'https://example.test/i1.jpg',
  thumbnail_url: 'https://example.test/i1-thumb.jpg',
  wear_count: 0,
  wears_since_wash: 0,
  effective_wash_interval: 3,
} as unknown as Item


const fixture = { ...item, name: 'Fixture shirt', favorite: false, is_archived: false }
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(fetch).mockReset()
  vi.mocked(api.get).mockResolvedValue([])
  vi.spyOn(console, 'error').mockImplementation(() => {})
})
afterEach(() => vi.restoreAllMocks())

async function chooseUpload() {
  const file = new File(['synthetic image for component test'], 'isolated-shirt.png', { type: 'image/png' })
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  expect(input).toBeTruthy()
  fireEvent.change(input, { target: { files: [file] } })
  await screen.findByAltText('Preview')
  return file
}

describe('lesson63 wardrobe UI', () => {
  it('blocks submission without an image', () => {
    render(<AddItemDialog open onOpenChange={vi.fn()} />, { wrapper: wrapperFor('en') })
    const submit = screen.getByRole('button', { name: 'Add Item' })
    expect(submit).toBeDisabled()
    fireEvent.click(submit)
    expect(fetch).not.toHaveBeenCalled()
  })

  it('submits multipart image and form fields to the API proxy', async () => {
    const close = vi.fn()
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify(fixture), { status: 201 }))
    render(<AddItemDialog open onOpenChange={close} />, { wrapper: wrapperFor('en') })
    const file = await chooseUpload()
    fireEvent.change(screen.getByLabelText('Name (optional)'), { target: { value: 'Isolated shirt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add Item' }))
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(fetch).toHaveBeenCalledTimes(1)
    const [url, request] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/v1/items')
    expect(request?.method).toBe('POST')
    const body = request?.body as FormData
    expect((body.get('image') as File).name).toBe(file.name)
    expect(body.get('name')).toBe('Isolated shirt')
  })

  it('shows authoritative item fields without mutating them', () => {
    render(<ItemDetailDialog item={fixture} open onOpenChange={vi.fn()} />, { wrapper: wrapperFor('en') })
    expect(screen.getByText('Fixture shirt')).toBeInTheDocument()
    expect(api.patch).not.toHaveBeenCalled()
    expect(api.delete).not.toHaveBeenCalled()
  })

  it('sends edited fields with the displayed item identifier', async () => {
    vi.mocked(api.patch).mockResolvedValueOnce({ ...fixture, name: 'Edited shirt' })
    render(<ItemDetailDialog item={fixture} open onOpenChange={vi.fn()} />, { wrapper: wrapperFor('en') })
    fireEvent.click(screen.getByRole('button', { name: 'Edit item' }))
    fireEvent.change(screen.getByPlaceholderText('Item name'), { target: { value: 'Edited shirt' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(api.patch).toHaveBeenCalledWith('/items/i1', expect.objectContaining({ name: 'Edited shirt', type: 'shirt' })))
    await waitFor(() => expect(screen.queryByPlaceholderText('Item name')).not.toBeInTheDocument())
  })

  it('requires confirmation before deleting and reports success', async () => {
    const close = vi.fn()
    vi.mocked(api.delete).mockResolvedValueOnce(undefined)
    render(<ItemDetailDialog item={fixture} open onOpenChange={close} />, { wrapper: wrapperFor('en') })
    fireEvent.click(screen.getByRole('button', { name: 'Delete item' }))
    expect(api.delete).not.toHaveBeenCalled()
    expect(screen.getByRole('alertdialog')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /^Delete$/ }))
    await waitFor(() => expect(api.delete).toHaveBeenCalledWith('/items/i1'))
    await waitFor(() => expect(close).toHaveBeenCalledWith(false))
    expect(toast.success).toHaveBeenCalled()
  })

  it('disables uploading when backend capabilities deny writes', async () => {
    render(
      <ApplicationAccessContext.Provider value={{ businessWrite: false, reason: 'Read only' }}>
        <AddItemDialog open onOpenChange={vi.fn()} />
      </ApplicationAccessContext.Provider>,
      { wrapper: wrapperFor('en') },
    )
    await chooseUpload()
    expect(screen.getByRole('button', { name: 'Add Item' })).toBeDisabled()
    expect(fetch).not.toHaveBeenCalled()
  })

  it('reports a rejected upload to the user and keeps the form open', async () => {
    const close = vi.fn()
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Isolated upload rejected' }), { status: 400 }))
    render(<AddItemDialog open onOpenChange={close} />, { wrapper: wrapperFor('en') })
    await chooseUpload()
    fireEvent.click(screen.getByRole('button', { name: 'Add Item' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(messages('en', 'wardrobe').addItem.uploadError))
    expect(close).not.toHaveBeenCalled()
    expect(screen.getByAltText('Preview')).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(console.error).not.toHaveBeenCalled()
  })
  it('keeps upload data and shows a safe error on network failure without retry', async () => {
    const close = vi.fn()
    vi.mocked(fetch).mockRejectedValueOnce(new Error('http://private.invalid/image?token=secret-test'))
    render(<AddItemDialog open onOpenChange={close} />, { wrapper: wrapperFor('en') })
    await chooseUpload()
    fireEvent.change(screen.getByLabelText('Name (optional)'), { target: { value: 'Keep this name' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add Item' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(messages('en', 'wardrobe').addItem.uploadError))
    expect(screen.getByLabelText('Name (optional)')).toHaveValue('Keep this name')
    expect(screen.getByAltText('Preview')).toBeInTheDocument()
    expect(close).not.toHaveBeenCalled()
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(console.error).not.toHaveBeenCalled()
  })
})
