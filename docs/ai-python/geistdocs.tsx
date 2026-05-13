export const Logo = () => (
  <span className="font-semibold text-gray-1000 text-lg leading-none tracking-[-3%]">
    AI SDK for Python
  </span>
);

export const github = {
  owner: "vercel-labs",
  repo: "ai-python",
};

export const nav = [
  {
    label: "Docs",
    href: "/docs",
  },
  {
    label: "Source",
    href: `https://github.com/${github.owner}/${github.repo}/`,
  },
];

export const suggestions = [
  "What is Geistdocs?",
  "What can I make with Geistdocs?",
  "What syntax does Geistdocs support?",
  "How do I deploy my Geistdocs site?",
];

export const title = "AI SDK for Python Documentation";

export const prompt =
  "You are a helpful assistant specializing in answering questions about the AI SDK for Python, toolkit for building LLM-powered applications and agent loops."
export const translations = {
  en: {
    displayName: "English",
  },
};

export const basePath: string | undefined = undefined;

/**
 * Unique identifier for this site, used in markdown request tracking analytics.
 * Each site using geistdocs should set this to a unique value (e.g. "ai-sdk-docs", "next-docs").
 */
export const siteId: string | undefined = undefined;
