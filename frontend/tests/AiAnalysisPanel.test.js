import { mount } from '@vue/test-utils';
import AiAnalysisPanel from '../src/components/AiAnalysisPanel.vue';

const baseAnalysis = {
  status: 'completed',
  severity_label: 'Moderate',
  recommendation: 'Investigate',
  confidence_score: '0.7',
  fraud_score: '0.44',
  needs_human_review: false,
  policy_findings: [
    {
      clause_id: 'CL-001',
      clause_type: 'coverage',
      source_citation: 'section 3.1',
      text: 'Comprehensive coverage applies to accidental body damage.',
      retrieval_breakdown: { dense_rank: 1, dense_contribution: 0.049, sparse_rank: null, sparse_contribution: 0 },
    },
  ],
  detections: [
    {
      class_name: 'dent',
      confidence: 0.82,
      bbox: [10, 10, 50, 50],
      saliency: {
        grid_size: 2,
        importance: [[1, 0], [0, 0]],
        peak_cell: { row: 0, col: 0 },
        peak_confidence_drop: 0.77,
      },
    },
  ],
  report_json: {
    damage_table: [{ class: 'dent', severity: 'Moderate', confidence: 0.82 }],
    next_steps: [],
    recommendation_reason: 'Coverage clause CL-001 applies.',
    severity_summary: {
      overall_severity: 'Moderate',
      severity_score: 0.12,
      per_region: [{ class_name: 'dent', confidence: 0.82, severity: 'Moderate' }],
    },
    consistency_check: {
      all_passed: false,
      checks: [
        { rule: 'citations_grounded', passed: true, detail: 'Every cited clause_id exists in the retrieved policy_findings.' },
        { rule: 'sub_limit_respected', passed: true, detail: 'No policy-limit violation.' },
        { rule: 'coverage_basis_present', passed: true, detail: 'A coverage clause supports the recommendation.' },
        { rule: 'fraud_signals_reflected', passed: false, detail: 'Fraud assessment flagged this claim but recommendation is still Approve.' },
      ],
    },
    fraud_assessment: {
      reason: 'Claimant name does not match policyholder of record',
      signals: ["Claimant name 'John Smith' does not match policyholder of record 'Jane Doe'"],
      needs_investigation: true,
      score_breakdown: [
        { label: 'Base risk from severity and claim amount', detail: 'Detected-damage severity score (0.09) plus a fixed baseline risk of 0.10', triggered: true, delta: 0.19, running_total: 0.19 },
        { label: 'Claimant/policyholder name mismatch', detail: "Claimant 'John Smith' vs policyholder of record 'Jane Doe'", triggered: true, delta: 0.25, running_total: 0.44 },
        { label: 'Policy inactive or expired at incident date', detail: "Policy status 'active'", triggered: false, delta: 0, running_total: 0.44 },
      ],
      narrative_analysis: {
        flags: [{ quote: 'needed the money fast', concern: 'financial pressure' }],
        available: true,
        reason: null,
      },
    },
  },
};

describe('AiAnalysisPanel explainability', () => {
  it('renders nothing extra when there is no explainability data', () => {
    const wrapper = mount(AiAnalysisPanel, {
      props: { analysis: { status: 'completed', policy_findings: [], report_json: { damage_table: [], next_steps: [] } } },
    });
    expect(wrapper.text()).not.toContain('AI explainability');
  });

  it('renders the damage saliency heatmap with its peak-region caption', () => {
    const wrapper = mount(AiAnalysisPanel, { props: { analysis: baseAnalysis } });
    expect(wrapper.text()).toContain('AI explainability');
    expect(wrapper.text()).toContain('Damage classification');
    expect(wrapper.find('.saliency-grid').exists()).toBe(true);
    expect(wrapper.findAll('.saliency-cell')).toHaveLength(4);
    expect(wrapper.text()).toContain('row 1, column 1');
    expect(wrapper.text()).toContain('77%');
  });

  it('renders the fraud score breakdown as a waterfall with triggered/untriggered steps', () => {
    const wrapper = mount(AiAnalysisPanel, { props: { analysis: baseAnalysis } });
    expect(wrapper.text()).toContain('Fraud score');
    expect(wrapper.text()).toContain('Claimant/policyholder name mismatch');
    expect(wrapper.text()).toContain('+0.25');
    expect(wrapper.find('.step-row-inactive').exists()).toBe(true);
    expect(wrapper.text()).toContain('not triggered');
  });

  it('renders the severity scoring breakdown per detected region', () => {
    const wrapper = mount(AiAnalysisPanel, { props: { analysis: baseAnalysis } });
    expect(wrapper.text()).toContain('Severity scoring');
    expect(wrapper.text()).toContain('12%');
    expect(wrapper.text()).toContain('dent');
    expect(wrapper.text()).toContain('confidence 82%');
  });

  it('renders grounded narrative red flags as quoted blockquotes', () => {
    const wrapper = mount(AiAnalysisPanel, { props: { analysis: baseAnalysis } });
    expect(wrapper.find('.narrative-flag').exists()).toBe(true);
    expect(wrapper.text()).toContain('needed the money fast');
    expect(wrapper.text()).toContain('financial pressure');
  });

  it('renders the consistency check with a failing rule surfaced', () => {
    const wrapper = mount(AiAnalysisPanel, { props: { analysis: baseAnalysis } });
    expect(wrapper.text()).toContain('Issue found');
    expect(wrapper.text()).toContain('Fraud signals are reflected in the recommendation');
    expect(wrapper.find('.consistency-fail').exists()).toBe(true);
  });

  it('shows retrieval provenance under each cited policy clause', () => {
    const wrapper = mount(AiAnalysisPanel, { props: { analysis: baseAnalysis } });
    expect(wrapper.text()).toContain('Retrieved by semantic match #1');
  });
});
