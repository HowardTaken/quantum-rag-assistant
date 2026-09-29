import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// Without vitest's `globals: true`, @testing-library/react's auto-cleanup (which relies
// on global afterEach) never registers, so each test's rendered DOM piles up across the
// file and later queries start matching leftover elements from earlier tests.
afterEach(() => {
  cleanup();
});

// jsdom doesn't implement scrollIntoView; App.tsx calls it to autoscroll the chat.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
