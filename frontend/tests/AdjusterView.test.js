import { flushPromises, mount } from '@vue/test-utils';
import { vi } from 'vitest';
import AdjusterView from '../src/views/AdjusterView.vue';
import * as api from '../src/services/api';

vi.mock('../src/services/api', () => ({
  getAdjusterDashboard: vi.fn(),
  getClaimDetail: vi.fn(),
  submitDecision: vi.fn(),
  annotatedPhotoUrl: vi.fn((claimId) => `/mock/${claimId}/annotated-photo`),
}));

const dashboardResponse = {
  summary: { pending_count: 1, approved_count: 2, denied_count: 0 },
  claims: [{ claim_id: 'CLM-1001', claimant_name: 'Ada Lovelace', status: 'submitted', claimed_amount: '1200' }],
};

const detailResponse = {
  claim_id: 'CLM-1001',
  status: 'under review',
  analysis_result: {
    status: 'completed',
    severity_label: 'Minor',
    recommendation: 'Approve',
    confidence_score: '0.9',
    fraud_score: '0.2',
    policy_findings: [],
    report_json: { damage_table: [], next_steps: [] },
    needs_human_review: false,
  },
};

describe('AdjusterView', () => {
  beforeEach(() => {
    api.getAdjusterDashboard.mockResolvedValue(dashboardResponse);
    api.getClaimDetail.mockResolvedValue(detailResponse);
    api.submitDecision.mockResolvedValue({ decision: 'approved' });
  });

  it('renders dashboard summary counts', async () => {
    const wrapper = mount(AdjusterView);
    await flushPromises();
    expect(wrapper.text()).toContain('CLM-1001');
    expect(wrapper.text()).toContain('Ada Lovelace');
  });

  it('loads and displays the AI analysis panel when a claim is reviewed', async () => {
    const wrapper = mount(AdjusterView);
    await flushPromises();

    const row = wrapper.find('tbody tr');
    await row.trigger('click');
    await flushPromises();

    expect(api.getClaimDetail).toHaveBeenCalledWith('CLM-1001');
    expect(wrapper.text()).toContain('Severity: Minor');
    expect(wrapper.text()).toContain('Recommendation: Approve');
  });
});
