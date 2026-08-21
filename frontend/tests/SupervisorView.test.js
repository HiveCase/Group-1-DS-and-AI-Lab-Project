import { flushPromises, mount } from '@vue/test-utils';
import { vi } from 'vitest';
import SupervisorView from '../src/views/SupervisorView.vue';
import * as api from '../src/services/api';

vi.mock('../src/services/api', () => ({
  getSupervisorAnalytics: vi.fn(),
}));

const analyticsResponse = {
  summary: {
    total_claims: 10,
    pending_count: 2,
    approved_count: 5,
    denied_count: 3,
    average_fraud_score: 0.42,
    severity_counts: { Minor: 5, Moderate: 3, Severe: 2 },
    coverage_flag_rate: 0.2,
    claims_processed_today: 4,
    system_status: {
      pipeline_status: 'operational',
      avg_analysis_time_seconds: 6.5,
      claims_awaiting_analysis: 1,
      recent_failure_count: 0,
    },
  },
};

describe('SupervisorView', () => {
  beforeEach(() => {
    api.getSupervisorAnalytics.mockResolvedValue(analyticsResponse);
  });

  it('renders KPI cards, severity distribution, and AI pipeline system status', async () => {
    const wrapper = mount(SupervisorView);
    await flushPromises();

    expect(wrapper.text()).toContain('Total claims');
    expect(wrapper.text()).toContain('10');
    expect(wrapper.text()).toContain('Severity distribution');
    expect(wrapper.text()).toContain('Minor');
    expect(wrapper.text()).toContain('operational');
    expect(wrapper.text()).toContain('Avg analysis time: 6.5s');
  });

  it('shows an empty state when there is no severity data yet', async () => {
    api.getSupervisorAnalytics.mockResolvedValue({
      summary: { ...analyticsResponse.summary, severity_counts: { Minor: 0, Moderate: 0, Severe: 0 } },
    });
    const wrapper = mount(SupervisorView);
    await flushPromises();

    expect(wrapper.text()).toContain('No completed AI analyses yet');
  });
});
