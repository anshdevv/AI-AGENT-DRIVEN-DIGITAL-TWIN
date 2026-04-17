import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the medical concierge shell", () => {
  render(<App />);

  expect(screen.getByText(/Medical Concierge/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Messages/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Diagnostics/i })).toBeInTheDocument();
});
