/**
 * Protocol that all Respan instrumentation plugins must implement.
 * Matches Python's Instrumentation Protocol.
 */
export interface RespanInstrumentation {
  name: string;
  activate(): void | Promise<void>;
  deactivate(): void | Promise<void>;
}
