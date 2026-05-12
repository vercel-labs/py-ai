import { createSearchRoute } from "@vercel/geistdocs/routes/search";
import { config } from "@/lib/geistdocs/config";
import { source } from "@/lib/geistdocs/source";

export const GET = createSearchRoute({ config, source });
