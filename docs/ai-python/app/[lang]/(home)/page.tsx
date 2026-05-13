import type { Metadata } from "next";
import {
  CommandPromptContent,
  CommandPromptCopy,
  CommandPromptList,
  CommandPromptPrefix,
  CommandPromptRoot,
  CommandPromptSurface,
  CommandPromptTrigger,
  CommandPromptTriggerDivider,
  CommandPromptViewport,
} from "@/components/ui/command-prompt";
import { CenteredSection } from "./components/centered-section";
import { CTA } from "./components/cta";
import { Hero } from "./components/hero";
import { OneTwoSection } from "./components/one-two-section";
import { Templates } from "./components/templates";
import { TextGridSection } from "./components/text-grid-section";

const title = "AI SDK for Python";
const description =
  "A toolkit for building LLM-powered applications and agent loops.";

export const metadata: Metadata = {
  title,
  description,
};

const templates = [
  {
    title: "Template 1",
    description: "Description of template 1",
    link: "https://example.com/template-1",
    image: "https://placehold.co/600x400.png",
  },
  {
    title: "Template 2",
    description: "Description of template 2",
    link: "https://example.com/template-2",
    image: "https://placehold.co/600x400.png",
  },
  {
    title: "Template 3",
    description: "Description of template 3",
    link: "https://example.com/template-3",
    image: "https://placehold.co/600x400.png",
  },
];

const textGridSection = [
  {
    id: "1",
    title: "Text Grid Section",
    description: "Description of text grid section",
  },
  {
    id: "2",
    title: "Text Grid Section",
    description: "Description of text grid section",
  },
  {
    id: "3",
    title: "Text Grid Section",
    description: "Description of text grid section",
  },
];

const COMMAND_FOR_HUMANS = "npx @vercel/geistdocs init";
const COMMAND_FOR_AGENTS = "npx @vercel/geistdocs init --agent";

const HomePage = () => (
  <div className="container mx-auto max-w-5xl">
    <Hero
      badge="Alpha is out now"
      description={description}
      title={title}
    >
      <CommandPromptRoot defaultValue="humans">
        <CommandPromptList>
          <CommandPromptTrigger className="min-w-[90px]" value="humans">
            For humans
          </CommandPromptTrigger>
          <CommandPromptTriggerDivider />
          <CommandPromptTrigger className="min-w-[84px]" value="agents">
            For agents
          </CommandPromptTrigger>
        </CommandPromptList>
        <CommandPromptSurface>
          <CommandPromptPrefix>$</CommandPromptPrefix>
          <CommandPromptViewport>
            <CommandPromptContent copyValue={COMMAND_FOR_HUMANS} value="humans">
              {COMMAND_FOR_HUMANS}
            </CommandPromptContent>
            <CommandPromptContent copyValue={COMMAND_FOR_AGENTS} value="agents">
              {COMMAND_FOR_AGENTS}
            </CommandPromptContent>
          </CommandPromptViewport>
          <CommandPromptCopy />
        </CommandPromptSurface>
      </CommandPromptRoot>
    </Hero>
    <div className="grid divide-y border-y sm:border-x">
      <TextGridSection data={textGridSection} />
      <CenteredSection
        description="Description of centered section"
        title="Centered Section"
      >
        <div className="aspect-video rounded-lg border bg-background" />
      </CenteredSection>
      <OneTwoSection
        description="Description of one/two section"
        title="One/Two Section"
      >
        <div className="aspect-video rounded-lg border bg-background" />
      </OneTwoSection>
      <Templates
        data={templates}
        description="See Geistdocs in action with one of our templates."
        title="Get started quickly"
      />
      <CTA cta="Get started" href="/docs" title="Start your docs today" />
    </div>
  </div>
);

export default HomePage;
