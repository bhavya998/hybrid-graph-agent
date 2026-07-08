"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Boxes, Activity } from "lucide-react";
import { checkHealth } from "@/lib/api";

export function Header() {
  const pathname = usePathname();
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      const ok = await checkHealth();
      if (active) setHealthy(ok);
    };
    poll();
    const interval = setInterval(poll, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { href: "/", label: "Chat" },
    { href: "/architecture", label: "Architecture" },
  ];

  return (
    <header className="flex items-center justify-between border-b border-border bg-surface/80 px-4 py-3 backdrop-blur-sm">
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-subtle">
          <Boxes className="h-4.5 w-4.5 text-accent" />
        </div>
        <div className="flex flex-col leading-none">
          <span className="text-sm font-semibold">hga</span>
          <span className="text-[10px] text-foreground-subtle">
            Hybrid Graph Agent
          </span>
        </div>
      </div>

      <nav className="flex items-center gap-1">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              pathname === item.href
                ? "bg-surface-3 text-foreground"
                : "text-foreground-muted hover:text-foreground hover:bg-surface-2"
            }`}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="flex items-center gap-1.5">
        <span
          className={`h-2 w-2 rounded-full ${
            healthy === null
              ? "bg-foreground-subtle"
              : healthy
                ? "bg-success"
                : "bg-error"
          }`}
        />
        <span className="text-[11px] text-foreground-subtle">
          {healthy === null ? "checking" : healthy ? "API online" : "API offline"}
        </span>
        <Activity className="h-3 w-3 text-foreground-subtle" />
      </div>
    </header>
  );
}
