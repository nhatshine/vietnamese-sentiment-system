import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { EnsembleExplanation } from './EnsembleExplanation';
import { parseModelInfo } from '../services/contracts';
import metadata from '../test/fixtures/model-info.json';

describe('current deployment explanation', () => {
  it('shows the backend model version and 70/30 weights without a stale statistical claim', () => {
    const info = parseModelInfo({ ...metadata, selected_alpha: .7,
      models: { ...metadata.models, transformer: { name: 'Transformer tự xây dựng · epoch 21' } } });
    render(<EnsembleExplanation response={null} info={info} />);
    expect(screen.getByText('70,00%')).toBeInTheDocument();
    expect(screen.getByText('30,00%')).toBeInTheDocument();
    expect(screen.getByText('Transformer tự xây dựng · epoch 21')).toBeInTheDocument();
    expect(screen.queryByText(/Mức chênh lệch F1 chưa được kiểm định/)).not.toBeInTheDocument();
  });
});
