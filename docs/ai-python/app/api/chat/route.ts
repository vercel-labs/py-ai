import { createChatRoute } from "@vercel/geistdocs/routes/chat";
import { config } from "@/lib/geistdocs/config";
import { geistdocsSource } from "@/lib/geistdocs/source";

export const { POST, maxDuration } = createChatRoute({
  config,
  source: geistdocsSource,
});
