import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import type { ToolUIPart } from "ai";
import { Fragment } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputSubmit,
} from "@/components/ai-elements/prompt-input";
import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { TooltipProvider } from "@/components/ui/tooltip";

export default function App() {
  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
    }),
  });

  const isLoading = status === "submitted" || status === "streaming";

  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col bg-background">
        <header className="border-b px-4 py-3">
          <div className="mx-auto w-full max-w-3xl">
            <h1 className="text-lg font-semibold">Python AI SDK Chat Demo</h1>
            <p className="text-sm text-muted-foreground">
              Powered by vercel-ai-sdk + FastAPI
            </p>
          </div>
        </header>

        <Conversation className="flex-1">
          <ConversationContent>
            <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-4">
              {messages.length === 0 ? (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <p>Send a message to start chatting</p>
                </div>
              ) : (
                messages.map((message) => (
                  <Fragment key={message.id}>
                    {message.parts.map((part, partIndex) => {
                      // Handle tool parts (type starts with "tool-")
                      if (part.type.startsWith("tool-")) {
                        const toolPart = part as ToolUIPart;
                        const isComplete = toolPart.state === "output-available";

                        return (
                          <Tool
                            key={`${message.id}-${partIndex}`}
                            defaultOpen={isComplete}
                          >
                            <ToolHeader
                              type={toolPart.type}
                              state={toolPart.state}
                            />
                            <ToolContent>
                              <ToolInput input={toolPart.input} />
                              <ToolOutput
                                output={toolPart.output}
                                errorText={toolPart.errorText}
                              />
                            </ToolContent>
                          </Tool>
                        );
                      }

                      // Handle text parts
                        if (part.type === "text") {
                        return (
                          <Message
                            key={`${message.id}-${partIndex}`}
                            from={message.role}
                          >
                            <MessageContent>
                              <MessageResponse>{part.text}</MessageResponse>
                            </MessageContent>
                          </Message>
                        );
                      }

                      return null;
                    })}
                  </Fragment>
                ))
              )}
            </div>
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <div className="border-t p-4">
          <div className="mx-auto w-full max-w-3xl">
            <PromptInput
              onSubmit={({ text }) => {
                if (text.trim()) {
                  sendMessage({ text });
                }
              }}
            >
              <PromptInputTextarea
                placeholder="Ask me anything..."
                disabled={isLoading}
              />
              <PromptInputFooter>
                <div />
                <PromptInputSubmit status={status} onStop={stop} />
              </PromptInputFooter>
            </PromptInput>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
