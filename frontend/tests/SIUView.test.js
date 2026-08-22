import { flushPromises, mount } from '@vue/test-utils';
import { vi } from 'vitest';
import SIUView from '../src/views/SIUView.vue';
import * as api from '../src/services/api';

vi.mock('../src/services/api', () => ({
  getSIUDashboard: vi.fn(),
  getClaimDetail: vi.fn(),
  submitSIUAction: vi.fn(),
}));

const dashboardResponse = {
  summary: { flagged_count: 1, under_investigation_count: 0, confirmed_fraud_count: 0 },
  claims: [{
    claim_id: 'CLM-2001',
    claimant_name: 'Grace Hopper',
    claim_type: 'Collision damage',
    claimed_amount: '4000',
    fraud_score: '0.8',
    needs_human_review: true,
    investigation_status: 'not_started',
    referral_reason: 'fraud_score',
  }],
};

const detailResponse = {
  claim_id: 'CLM-2001',
  status: 'under review',
  analysis_result: {
    status: 'completed',
    severity_label: 'Severe',
    recommendation: 'Investigate',
    confidence_score: '0.55',
    fraud_score: '0.8',
    policy_findings: [],
    report_json: { damage_table: [], next_steps: [], fraud_assessment: { reason: 'High severity and claim amount suggest deeper review' } },
    needs_human_review: true,
  },
};

describe('SIUView', () => {
  beforeEach(() => {
    api.getSIUDashboard.mockResolvedValue(dashboardResponse);
    api.getClaimDetail.mockResolvedValue(detailResponse);
    api.submitSIUAction.mockResolvedValue({ claim_id: 'CLM-2001', status: 'under_investigation' });
  });

  it('renders high-risk claims from the SIU dashboard', async () => {
    const wrapper = mount(SIUView);
    await flushPromises();
    expect(wrapper.text()).toContain('CLM-2001');
    expect(wrapper.text()).toContain('Grace Hopper');
  });

  it('opens a claim, reuses the AI analysis panel, and records an investigation action', async () => {
    const wrapper = mount(SIUView);
    await flushPromises();

    await wrapper.find('button').trigger('click');
    await flushPromises();

    expect(api.getClaimDetail).toHaveBeenCalledWith('CLM-2001');
    expect(wrapper.text()).toContain('Fraud factors');
    expect(wrapper.text()).toContain('High severity and claim amount suggest deeper review');

    const openInvestigationButton = wrapper.findAll('button').find((b) => b.text() === 'Open investigation');
    await openInvestigationButton.trigger('click');
    await flushPromises();

    expect(api.submitSIUAction).toHaveBeenCalledWith('CLM-2001', expect.objectContaining({ status: 'under_investigation' }));
  });
});
