import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const raw = readFileSync(resolve(root, 'src/data/dashboardData.json'), 'utf8')
const data = JSON.parse(raw)
const expect = (condition, message) => { if (!condition) throw new Error(message) }

expect(data.meta.commit === '4dab398a82b399076b7d201009ea9ab3bdc7909a', 'Frozen commit mismatch')
expect(data.official.sessions.length === 200, 'Official session count must equal 200')
expect(data.official.runs[1].hr === 1, 'Version A HR mismatch')
expect(data.official.runs[1].mrr === 0.936429, 'Version A MRR mismatch')
expect(data.official.runs[1].score === 0.954329, 'Version A TechnicalScore mismatch')
expect(data.official.runs[0].hr === 0.125, 'Weak baseline HR mismatch')
expect(data.robustness.absolute.version_a.synonym.exact.hr === 0, 'Synonym HR mismatch')
expect(data.engineering.tokens === 0 && data.engineering.apiCost === 0, 'Cost mismatch')
expect(!raw.includes('/Users/') && !raw.includes('API_KEY') && !raw.includes('token='), 'Private path or secret marker found')
console.log('Dashboard data validation passed: sources reconciled, 200 sessions, no private paths or secret markers')
