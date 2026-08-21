import { mount } from '@vue/test-utils';
import ClaimantView from '../src/views/ClaimantView.vue';

describe('ClaimantView', () => {
  it('renders the claimant form', () => {
    const wrapper = mount(ClaimantView);
    expect(wrapper.text()).toContain('Claimant portal');
  });
});
