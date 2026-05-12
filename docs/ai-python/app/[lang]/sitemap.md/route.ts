import { createSitemapMarkdownRoute } from "@vercel/geistdocs/routes/sitemap";
import { config } from "@/lib/geistdocs/config";
import { source } from "@/lib/geistdocs/source";

export const { GET, generateStaticParams, revalidate, dynamic } =
  createSitemapMarkdownRoute({
    config,
    source,
  });
