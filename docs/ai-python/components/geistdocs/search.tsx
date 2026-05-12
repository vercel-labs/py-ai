"use client";

import { SearchButton as PackageSearchButton } from "@vercel/geistdocs/controls";
import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

type SearchButtonProps = ComponentProps<typeof PackageSearchButton>;

export const SearchButton = ({ className, ...props }: SearchButtonProps) => (
  <PackageSearchButton
    className={cn(
      "group h-10 justify-between gap-8 pr-1.5 font-normal text-muted-foreground shadow-none lg:h-8 lg:w-[150px] lg:bg-background-200 hover:lg:bg-background-200 [&_[data-slot=kbd]]:transition-colors group-hover:[&_[data-slot=kbd]]:text-gray-1000",
      className
    )}
    {...props}
  />
);
