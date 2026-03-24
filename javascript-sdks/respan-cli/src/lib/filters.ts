/**
 * Shared filter parser for CLI commands.
 *
 * Parses --filter strings in `field:operator:value` format into the API's
 * `{ field: { operator, value } }` body format.
 *
 * Syntax:  field:operator:value
 *   - field::value        → exact match (empty operator)
 *   - field:gt:value      → greater than
 *   - field:in:a,b,c      → value in list
 *   - field:isnull:true    → null check
 *
 * Numeric values are auto-detected and converted.
 * Multiple --filter flags are merged into a single filters object.
 */

const VALID_OPERATORS = new Set([
  '', 'not', 'lt', 'lte', 'gt', 'gte',
  'contains', 'icontains', 'startswith', 'endswith',
  'in', 'isnull', 'iexact',
]);

function coerceValue(v: string): string | number | boolean {
  if (v === 'true') return true;
  if (v === 'false') return false;
  const n = Number(v);
  if (!Number.isNaN(n) && v.trim() !== '') return n;
  return v;
}

export function parseFilters(
  filterStrings: string[],
): Record<string, { operator: string; value: unknown[] }> {
  const result: Record<string, { operator: string; value: unknown[] }> = {};

  for (const raw of filterStrings) {
    // Split on first two colons: field:operator:value
    // field::value means operator is empty string
    const firstColon = raw.indexOf(':');
    if (firstColon === -1) {
      throw new Error(`Invalid filter format: "${raw}". Expected field:operator:value (e.g. model::gpt-4 or cost:gt:0.01)`);
    }

    const field = raw.slice(0, firstColon);
    const rest = raw.slice(firstColon + 1);

    const secondColon = rest.indexOf(':');
    if (secondColon === -1) {
      throw new Error(`Invalid filter format: "${raw}". Expected field:operator:value (e.g. model::gpt-4 or cost:gt:0.01)`);
    }

    const operator = rest.slice(0, secondColon);
    const valueStr = rest.slice(secondColon + 1);

    if (!VALID_OPERATORS.has(operator)) {
      throw new Error(`Unknown filter operator: "${operator}". Valid operators: ${[...VALID_OPERATORS].filter(Boolean).join(', ')} (or empty for exact match)`);
    }

    if (!field) {
      throw new Error(`Filter field cannot be empty: "${raw}"`);
    }

    // For "in" operator, split on comma to create array
    let values: unknown[];
    if (operator === 'in') {
      values = valueStr.split(',').map((v) => coerceValue(v.trim()));
    } else {
      values = [coerceValue(valueStr)];
    }

    result[field] = { operator, value: values };
  }

  return result;
}

/** Filter syntax documentation for use in command descriptions */
export const FILTER_SYNTAX_HELP = `FILTER SYNTAX: field:operator:value

OPERATORS:
  (empty)     Exact match         model::gpt-4
  not         Not equal           status_code:not:200
  gt          Greater than        cost:gt:0.01
  gte         Greater than/equal  latency:gte:1.0
  lt          Less than           cost:lt:0.5
  lte         Less than/equal     prompt_tokens:lte:100
  contains    Contains substring  error_message:contains:timeout
  icontains   Case-insensitive    model:icontains:gpt
  startswith  Starts with         model:startswith:gpt
  endswith    Ends with           model:endswith:mini
  in          Value in list       model:in:gpt-4,gpt-4o
  isnull      Is null             error_message:isnull:true
  iexact      Case-insens. exact  status:iexact:success`;

export const LOG_FIELDS_HELP = `FILTERABLE FIELDS (logs):
  model, status_code, status, cost, latency, prompt_tokens,
  completion_tokens, customer_identifier, custom_identifier,
  thread_identifier, trace_unique_id, span_name, span_workflow_name,
  environment, log_type, error_message, failed, provider_id,
  deployment_name, prompt_name, prompt_id, unique_id, stream,
  temperature, max_tokens, tokens_per_second, time_to_first_token,
  total_request_tokens, metadata__<key>, scores__<evaluator_id>`;

export const TRACE_FIELDS_HELP = `FILTERABLE FIELDS (traces):
  trace_unique_id, customer_identifier, environment, span_count,
  llm_call_count, error_count, total_cost, total_tokens,
  total_prompt_tokens, total_completion_tokens, duration,
  span_workflow_name, metadata__<key>`;

export const FILTER_EXAMPLES = `EXAMPLES:
  --filter model::gpt-4o --filter cost:gt:0.01
  --filter status_code:not:200
  --filter metadata__env::production
  --filter model:in:gpt-4,gpt-4o`;
