import { render, screen } from '@testing-library/react';
import App from './App';

test('renders patient conversation on root', () => {
  render(<App />);
  expect(screen.getByText(/medical concierge is online/i)).toBeInTheDocument();
});
