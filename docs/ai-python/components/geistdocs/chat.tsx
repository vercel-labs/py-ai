import { AskAIButton } from "@vercel/geistdocs/controls";

interface ChatProps {
  basePath: string | undefined;
  suggestions: string[];
}

export const Chat = ({
  basePath: _basePath,
  suggestions: _suggestions,
}: ChatProps) => (
  <>
    <AskAIButton
      className="hidden shrink-0 shadow-none md:flex"
      size="sm"
      variant="outline"
    />
    <AskAIButton
      className="shadow-none md:hidden"
      size="sm"
      variant="outline"
    />
  </>
);
